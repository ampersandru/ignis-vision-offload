//! Per-model semantic binder (ADR 0002).
//!
//! Rust port of the reference `binder` (`ninfer` `binder.h`/`binder.cpp`):
//! owns the per-model semantic layer the generic reader deliberately leaves
//! out — object-consumption validation, placement decisions, and the
//! materialization plan.
//!
//! Contract (ADR 0002): the binder must consume *every* object of the
//! reader's directory — [`Binder::finish`] fails if any object is left
//! unconsumed or has no placement. An unconsumed object is a load failure.

use crate::{
    align_up, fail, checked_add, NumericFormat, Object, PayloadSpan, Reader, Result,
    ResourceEncoding, StorageLayout, DIRECT_IO_ALIGNMENT,
};

/// Where a tensor the kernels read lives while the model runs (ADR 0062).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Residence {
    /// In the device arena, in VRAM.
    #[default]
    Device,
    /// In mapped pinned host memory (`cudaHostAllocMapped`), read in place
    /// over the bus through its mapped device pointer: no VRAM.
    HostMapped,
}

/// How a device tensor is held in the arena (GitHub #341).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceEncoding {
    /// The stored payload, byte for byte.
    #[default]
    Stored,
    /// A BF16 matrix converted at load to the FP8 row-scale encoding
    /// ([`crate::fp8`]) in these row blocks: one payload per block, each at a
    /// multiple of 256 bytes (the 27B's BF16 exception tensors).
    Fp8Rows(&'static [u32]),
}

impl DeviceEncoding {
    /// The format and layout a tensor stored as `format` / `layout` is held
    /// as on the device under this encoding.
    pub fn held_as(self, format: NumericFormat, layout: StorageLayout) -> (NumericFormat, StorageLayout) {
        match self {
            Self::Stored => (format, layout),
            Self::Fp8Rows(_) => (NumericFormat::Fp8E4M3FnRowBf16S, StorageLayout::RowScaleV1),
        }
    }
}

/// A stable per-object handle (its directory index in `Reader::objects()`
/// order).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObjectHandle {
    /// Directory index of the object (its position in `Reader::objects()`).
    pub index: usize,
}

/// One device-object placement in the plan's arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DevicePlacement {
    /// The placed object.
    pub handle: ObjectHandle,
    /// Offset inside the device arena (alignment-aware, binder-computed).
    pub offset: u64,
    /// The bytes it takes in the arena: the stored payload's, or its
    /// conversion's ([`DevicePlacement::encoding`]).
    pub bytes: u64,
    /// File alignment applied to this object's offset (its layout's,
    /// [`crate::tensor_alignment`]).
    pub alignment: u64,
    /// How it is held: as stored, or converted at load (GitHub #341).
    pub encoding: DeviceEncoding,
}

/// One host-retained placement: its bytes stay in RAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostPlacement {
    /// The retained object.
    pub handle: ObjectHandle,
}

/// One placement in the **expert pool** (GLOSSARY.md: every expert
/// projection, in pinned host RAM for the life of the model; residency moves
/// them from there into its device cache, spec flash-next/03). The
/// materializer does not fill the pool; residency does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertPoolPlacement {
    pub handle: ObjectHandle,
    /// Offset inside the pool (aligned to the object's file alignment).
    pub offset: u64,
    pub bytes: u64,
    pub alignment: u64,
}

/// One placement in **mapped pinned host memory** (ADR 0062): a tensor the
/// kernels read in place through its mapped device pointer, so it takes no
/// VRAM -- the 27B's token embedding. The materializer reads the payload's
/// aligned file span straight into the allocation, once, at load: each span
/// starts at a [`DIRECT_IO_ALIGNMENT`]-aligned offset, so the read needs no
/// staging copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostMappedPlacement {
    pub handle: ObjectHandle,
    /// Where the aligned span starts inside the allocation.
    pub span_offset: u64,
    /// The span's absolute file offset (the payload's, aligned down).
    pub span_file_offset: u64,
    /// The span's length (the payload's end aligned up, less its start).
    pub span_bytes: u64,
    /// Where the tensor starts inside the allocation: past `span_offset` by
    /// as much as its payload is past `span_file_offset`.
    pub offset: u64,
    /// Exact stored payload length.
    pub bytes: u64,
}

/// One host-streamed object: read from the file on demand by its owner
/// (Flash-Next's n-gram table, spec flash-next/04). The binder hands over
/// its file range and nothing reads it at load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamedPlacement {
    pub handle: ObjectHandle,
    /// Absolute file offset of the payload.
    pub file_offset: u64,
    pub bytes: u64,
}

/// The binder's terminal output: placement decisions + device capacity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterializationPlan {
    /// Total directory objects (device + host placements).
    pub object_count: usize,
    /// Device arena size: the (offset + bytes) of the last device object,
    /// alignment-aware.
    pub device_capacity_bytes: u64,
    /// Device placements in binding order.
    pub device_objects: Vec<DevicePlacement>,
    /// Host-retained resources in binding order.
    pub host_objects: Vec<HostPlacement>,
    /// Host-retained tensors in binding order (small tensors the host reads,
    /// kept apart from the resources).
    pub host_tensor_objects: Vec<HostPlacement>,
    /// Expert pool size, alignment-aware (0 without pool objects).
    pub expert_pool_capacity_bytes: u64,
    /// Expert pool placements in binding order.
    pub expert_pool_objects: Vec<ExpertPoolPlacement>,
    /// Host-streamed objects in binding order.
    pub streamed_objects: Vec<StreamedPlacement>,
    /// The mapped pinned host allocation's size: the end of its last span (0
    /// without host-mapped objects).
    pub host_mapped_capacity_bytes: u64,
    /// Host-mapped placements in binding order (ADR 0062).
    pub host_mapped_objects: Vec<HostMappedPlacement>,
}

/// The per-model binder: consumes every directory object and produces the
/// [`MaterializationPlan`] (ADR 0002).
pub struct Binder<'a> {
    reader: &'a Reader,
    consumed: Vec<bool>,
    planned: Vec<bool>,
    device_objects: Vec<DevicePlacement>,
    host_objects: Vec<HostPlacement>,
    capacity: u64,
    host_tensor_objects: Vec<HostPlacement>,
    expert_pool_objects: Vec<ExpertPoolPlacement>,
    expert_pool_capacity: u64,
    streamed_objects: Vec<StreamedPlacement>,
    host_mapped_objects: Vec<HostMappedPlacement>,
    host_mapped_capacity: u64,
}

impl<'a> Binder<'a> {
    /// Bind against an opened reader. Every object must be consumed before
    /// [`Binder::finish`] succeeds.
    pub fn new(reader: &'a Reader) -> Self {
        let object_count = reader.objects().len();
        Self {
            reader,
            consumed: vec![false; object_count],
            planned: vec![false; object_count],
            device_objects: Vec::new(),
            host_objects: Vec::new(),
            capacity: 0,
            host_tensor_objects: Vec::new(),
            expert_pool_objects: Vec::new(),
            expert_pool_capacity: 0,
            streamed_objects: Vec::new(),
            host_mapped_objects: Vec::new(),
            host_mapped_capacity: 0,
        }
    }

    /// Consume a directory object by exact name, validating it against the
    /// stored descriptor (the required format + layout + shape must match
    /// the stored tensor descriptor exactly).
    pub fn require_tensor(
        &mut self,
        name: &str,
        format: NumericFormat,
        layout: StorageLayout,
        shape: &[u64],
    ) -> Result<ObjectHandle> {
        let handle = self.consume(name)?;
        match &self.reader.objects()[handle.index] {
            Object::Tensor(t)
                if t.format == format
                    && t.layout == layout
                    && t.shape.as_slice() == shape =>
            {
                Ok(handle)
            }
            Object::Tensor(_) => Err(fail(format!(
                "tensor descriptor does not match target contract: {name}"
            ))),
            Object::Resource(_) => Err(fail(format!(
                "required tensor is a resource: {name}"
            ))),
        }
    }

    /// Consume a tensor whose format may be any of `formats` (a family
    /// that stores a per-object parameter in its format code, like the
    /// trellis K): layout and shape must match exactly. Returns the format
    /// found.
    pub fn require_tensor_of(
        &mut self,
        name: &str,
        formats: &[NumericFormat],
        layout: StorageLayout,
        shape: &[u64],
    ) -> Result<(ObjectHandle, NumericFormat)> {
        let handle = self.consume(name)?;
        match &self.reader.objects()[handle.index] {
            Object::Tensor(t)
                if formats.contains(&t.format)
                    && t.layout == layout
                    && t.shape.as_slice() == shape =>
            {
                Ok((handle, t.format))
            }
            Object::Tensor(_) => Err(fail(format!(
                "tensor descriptor does not match target contract: {name}"
            ))),
            Object::Resource(_) => Err(fail(format!(
                "required tensor is a resource: {name}"
            ))),
        }
    }

    /// Consume a required resource by exact name, validating the encoding.
    pub fn require_resource(
        &mut self,
        name: &str,
        encoding: ResourceEncoding,
    ) -> Result<ObjectHandle> {
        let handle = self.consume(name)?;
        match &self.reader.objects()[handle.index] {
            Object::Resource(r) if r.encoding == encoding => Ok(handle),
            Object::Resource(_) => Err(fail(format!(
                "resource encoding does not match target contract: {name}"
            ))),
            Object::Tensor(_) => Err(fail(format!(
                "required resource is a tensor: {name}"
            ))),
        }
    }

    /// The directory descriptor of a consumed object.
    pub fn descriptor(&self, handle: ObjectHandle) -> &Object {
        &self.reader.objects()[handle.index]
    }

    /// The validated payload span of a consumed object (mapping-backed).
    pub fn payload(&self, handle: ObjectHandle) -> Result<PayloadSpan<'_>> {
        self.reader.payload_at(self.descriptor(handle))
    }

    /// Place a consumed tensor into the device arena (256-byte aligned).
    pub fn materialize_on_device(&mut self, handle: ObjectHandle) -> Result<()> {
        self.claim_placement(handle)?;
        let tensor = match &self.reader.objects()[handle.index] {
            Object::Tensor(t) => t,
            Object::Resource(_) => {
                return Err(fail(format!(
                    "resource cannot be materialized as a device tensor: {}",
                    self.reader.objects()[handle.index].name()
                )))
            }
        };
        let alignment = crate::tensor_alignment(tensor.layout);
        let offset = align_up(self.capacity, alignment, "materialization plan offset")?;
        let capacity = checked_add(offset, tensor.bytes, "materialization plan size")?;
        self.device_objects.push(DevicePlacement {
            handle,
            offset,
            bytes: tensor.bytes,
            alignment,
            encoding: DeviceEncoding::Stored,
        });
        self.capacity = capacity;
        self.planned[handle.index] = true;
        Ok(())
    }

    /// Place a consumed BF16 matrix into the device arena converted to the
    /// FP8 row-scale encoding in row `blocks` (GitHub #341): its slot is the
    /// conversion's bytes, 256-byte aligned. Refuses anything but a rank-2
    /// contiguous BF16 tensor whose columns are a multiple of 16 and whose
    /// rows the blocks cover exactly.
    pub fn materialize_on_device_as_fp8(&mut self, handle: ObjectHandle, blocks: &'static [u32]) -> Result<()> {
        self.claim_placement(handle)?;
        let tensor = match &self.reader.objects()[handle.index] {
            Object::Tensor(t) => t,
            Object::Resource(r) => return Err(fail(format!("resource cannot be converted to FP8: {}", r.name))),
        };
        let name = &tensor.name;
        if tensor.format != NumericFormat::Bf16 || tensor.layout != StorageLayout::ContiguousLeV1 || tensor.shape.len() != 2 {
            return Err(fail(format!("only a rank-2 contiguous BF16 tensor converts to FP8: {name}")));
        }
        let (rows, cols) = (tensor.shape[0], tensor.shape[1]);
        if cols == 0 || cols % 16 != 0 || cols > u64::from(u32::MAX) {
            return Err(fail(format!("an FP8 conversion's columns are a positive multiple of 16: {name}")));
        }
        if blocks.iter().map(|&b| u64::from(b)).sum::<u64>() != rows || blocks.contains(&0) {
            return Err(fail(format!("the FP8 row blocks {blocks:?} do not cover the {rows} rows of {name}")));
        }
        let bytes = crate::fp8::block_offsets(blocks, cols as u32)[blocks.len()];
        let offset = align_up(self.capacity, 256, "materialization plan offset")?;
        let capacity = checked_add(offset, bytes, "materialization plan size")?;
        self.device_objects.push(DevicePlacement {
            handle,
            offset,
            bytes,
            alignment: 256,
            encoding: DeviceEncoding::Fp8Rows(blocks),
        });
        self.capacity = capacity;
        self.planned[handle.index] = true;
        Ok(())
    }

    /// Retain a consumed resource on the host (its bytes stay in RAM; the
    /// reference `take_resource_bytes` pattern).
    pub fn retain_on_host(&mut self, handle: ObjectHandle) -> Result<()> {
        self.claim_placement(handle)?;
        if !matches!(&self.reader.objects()[handle.index], Object::Resource(_)) {
            return Err(fail(format!(
                "tensor cannot be retained as a host resource: {}",
                self.reader.objects()[handle.index].name()
            )));
        }
        self.host_objects.push(HostPlacement { handle });
        self.planned[handle.index] = true;
        Ok(())
    }

    /// Retain a consumed tensor's bytes on the host (a small tensor the
    /// host reads, e.g. Flash-Next's n-gram hash buffers).
    pub fn retain_tensor_on_host(&mut self, handle: ObjectHandle) -> Result<()> {
        self.claim_placement(handle)?;
        if !matches!(&self.reader.objects()[handle.index], Object::Tensor(_)) {
            return Err(fail(format!(
                "resource cannot be retained as a host tensor: {}",
                self.reader.objects()[handle.index].name()
            )));
        }
        self.host_tensor_objects.push(HostPlacement { handle });
        self.planned[handle.index] = true;
        Ok(())
    }

    /// Place a consumed tensor in the expert pool, aligned to its file
    /// alignment. Returns its pool offset.
    pub fn place_in_expert_pool(&mut self, handle: ObjectHandle) -> Result<u64> {
        self.claim_placement(handle)?;
        let tensor = match &self.reader.objects()[handle.index] {
            Object::Tensor(t) => t,
            Object::Resource(r) => {
                return Err(fail(format!(
                    "resource cannot be placed in the expert pool: {}",
                    r.name
                )))
            }
        };
        let alignment = crate::tensor_alignment(tensor.layout);
        let offset = align_up(self.expert_pool_capacity, alignment, "expert pool offset")?;
        self.expert_pool_capacity = checked_add(offset, tensor.bytes, "expert pool size")?;
        self.expert_pool_objects.push(ExpertPoolPlacement {
            handle,
            offset,
            bytes: tensor.bytes,
            alignment,
        });
        self.planned[handle.index] = true;
        Ok(offset)
    }

    /// Place a consumed tensor in mapped pinned host memory (ADR 0062): its
    /// aligned file span goes at the next aligned offset of the allocation.
    pub fn place_in_host_mapped(&mut self, handle: ObjectHandle) -> Result<()> {
        self.claim_placement(handle)?;
        if let Object::Resource(r) = &self.reader.objects()[handle.index] {
            return Err(fail(format!("resource cannot be placed in mapped host memory: {}", r.name)));
        }
        let (payload_offset, bytes) = {
            let span = self.payload(handle)?;
            (span.absolute_offset, span.data.len() as u64)
        };
        let span_file_offset = payload_offset & !(DIRECT_IO_ALIGNMENT - 1);
        let payload_end = checked_add(payload_offset, bytes, "host-mapped span end")?;
        let span_bytes = align_up(payload_end, DIRECT_IO_ALIGNMENT, "host-mapped span end")? - span_file_offset;
        let span_offset = align_up(self.host_mapped_capacity, DIRECT_IO_ALIGNMENT, "host-mapped span offset")?;
        self.host_mapped_capacity = checked_add(span_offset, span_bytes, "host-mapped allocation size")?;
        self.host_mapped_objects.push(HostMappedPlacement {
            handle,
            span_offset,
            span_file_offset,
            span_bytes,
            offset: span_offset + (payload_offset - span_file_offset),
            bytes,
        });
        self.planned[handle.index] = true;
        Ok(())
    }

    /// Hand a consumed object's file range to its owner instead of loading
    /// it (host-streamed: nothing reads its payload at load).
    pub fn stream_from_host(&mut self, handle: ObjectHandle) -> Result<()> {
        self.claim_placement(handle)?;
        let span = self.payload(handle)?;
        let placement = StreamedPlacement {
            handle,
            file_offset: span.absolute_offset,
            bytes: span.data.len() as u64,
        };
        self.streamed_objects.push(placement);
        self.planned[handle.index] = true;
        Ok(())
    }

    /// Refuse a second placement of the same object.
    fn claim_placement(&self, handle: ObjectHandle) -> Result<()> {
        if self.planned[handle.index] {
            return Err(fail(format!(
                "artifact object has more than one materialization placement: {}",
                self.reader.objects()[handle.index].name()
            )));
        }
        Ok(())
    }

    /// Mark a consumed object as validated-only (no placement, no upload).
    pub fn validate_only(&mut self, handle: ObjectHandle) -> Result<()> {
        self.claim_placement(handle)?;
        self.planned[handle.index] = true;
        Ok(())
    }

    /// Build a plan from the placements made so far, without requiring
    /// every directory object to be consumed.
    ///
    /// For a binder scoped to less than the whole directory (e.g. the
    /// text-scope model load, which leaves vision / MTP / frontend objects
    /// untouched) — [`Binder::finish`] stays the ADR 0002 contract for a
    /// binder that owns the whole artifact.
    pub fn plan(&self) -> MaterializationPlan {
        MaterializationPlan {
            object_count: self.consumed.iter().filter(|&&c| c).count(),
            device_capacity_bytes: self.capacity,
            device_objects: self.device_objects.clone(),
            host_objects: self.host_objects.clone(),
            host_tensor_objects: self.host_tensor_objects.clone(),
            expert_pool_capacity_bytes: self.expert_pool_capacity,
            expert_pool_objects: self.expert_pool_objects.clone(),
            streamed_objects: self.streamed_objects.clone(),
            host_mapped_capacity_bytes: self.host_mapped_capacity,
            host_mapped_objects: self.host_mapped_objects.clone(),
        }
    }

    /// Finish binding: every object consumed, every object placed.
    ///
    /// ADR 0002 — an unconsumed object is a load failure: `finish()` errors
    /// if any directory object was not consumed by an explicit
    /// `require_tensor` / `require_resource` call, or has no placement.
    pub fn finish(&self) -> Result<MaterializationPlan> {
        if let Some(i) = self.consumed.iter().position(|c| !c) {
            return Err(fail(format!(
                "artifact object was not consumed by the selected target: {}",
                self.reader.objects()[i].name()
            )));
        }
        if let Some(i) = self.planned.iter().position(|p| !p) {
            return Err(fail(format!(
                "artifact object has no materialization placement: {}",
                self.reader.objects()[i].name()
            )));
        }
        Ok(MaterializationPlan {
            object_count: self.consumed.len(),
            device_capacity_bytes: self.capacity,
            device_objects: self.device_objects.clone(),
            host_objects: self.host_objects.clone(),
            host_tensor_objects: self.host_tensor_objects.clone(),
            expert_pool_capacity_bytes: self.expert_pool_capacity,
            expert_pool_objects: self.expert_pool_objects.clone(),
            streamed_objects: self.streamed_objects.clone(),
            host_mapped_capacity_bytes: self.host_mapped_capacity,
            host_mapped_objects: self.host_mapped_objects.clone(),
        })
    }

    /// Consume a directory object by exact name (once — a second bind of the
    /// same object is an error).
    fn consume(&mut self, name: &str) -> Result<ObjectHandle> {
        let index = self
            .reader
            .index_of(name)
            .ok_or_else(|| fail(format!("required artifact object is missing: {name}")))?;
        if self.consumed[index] {
            return Err(fail(format!(
                "artifact object was bound more than once: {name}"
            )));
        }
        self.consumed[index] = true;
        Ok(ObjectHandle { index })
    }
}

// ---------------------------------------------------------------------------
// Tests (the binder is pure Rust — ADR 0002 failure paths, ADR 0006 stand-in)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture;

    /// A fixture reader (five objects: four tensors + one resource); the temp
    /// artifact is kept alive so the file persists for the reader's mapping.
    fn fixture_reader(tag: &str) -> (fixture::TempArtifact, Reader) {
        let objects = fixture::all_layout_objects();
        let payload = fixture::all_layout_payload();
        let artifact =
            fixture::write_fixture(&objects, &payload, tag).expect("write fixture");
        let reader = Reader::open(&artifact.path).expect("open fixture");
        (artifact, reader)
    }

    /// Consume + place the four tensor objects (the resource is left alone).
    fn bind_devices(binder: &mut Binder) {
        let bf16 = binder
            .require_tensor("w/bf16", NumericFormat::Bf16, StorageLayout::ContiguousLeV1, &[4, 8])
            .expect("bf16");
        binder.materialize_on_device(bf16).expect("place bf16");
        let nvfp4 = binder
            .require_tensor("w/nvfp4", NumericFormat::Nvfp4, StorageLayout::BlockScaleK16M128x4V1, &[128, 64])
            .expect("nvfp4");
        binder.materialize_on_device(nvfp4).expect("place nvfp4");
        let q4 = binder
            .require_tensor("w/q4", NumericFormat::Q4G64F16S, StorageLayout::RowSplitK128V1, &[128, 128])
            .expect("q4");
        binder.materialize_on_device(q4).expect("place q4");
        let fp8 = binder
            .require_tensor("w/fp8", NumericFormat::Fp8E4M3FnRowBf16S, StorageLayout::RowScaleV1, &[128, 64])
            .expect("fp8");
        binder.materialize_on_device(fp8).expect("place fp8");
    }

    /// Consume + place all five fixture objects (the ADR 0002 happy path).
    fn bind_all(binder: &mut Binder) {
        let resource = binder
            .require_resource("frontend/tokenizer.json", ResourceEncoding::RawBytesV1)
            .expect("resource");
        binder.retain_on_host(resource).expect("retain resource");
        bind_devices(binder);
    }

    /// GitHub #341: a BF16 matrix placed converted to FP8 takes its
    /// conversion's bytes at a 256-byte offset; anything else is refused.
    #[test]
    fn a_bf16_matrix_converts_to_fp8_row_blocks_and_nothing_else_does() {
        let objects = vec![
            fixture::FixtureObject::Tensor {
                name: "w/a",
                shape: vec![4, 8],
                format: "BF16",
                layout: "contiguous-le-v1",
                offset: 0,
                bytes: 64,
            },
            fixture::FixtureObject::Tensor {
                name: "w/b",
                shape: vec![48, 64],
                format: "BF16",
                layout: "contiguous-le-v1",
                offset: 256,
                bytes: 48 * 64 * 2,
            },
        ];
        let artifact = fixture::write_fixture(&objects, &vec![0u8; 256 + 48 * 64 * 2], "fp8-rows").expect("write");
        let reader = Reader::open(&artifact.path).expect("open");
        let mut binder = Binder::new(&reader);
        let a = binder.require_tensor("w/a", NumericFormat::Bf16, StorageLayout::ContiguousLeV1, &[4, 8]).expect("a");
        binder.materialize_on_device(a).expect("a stored");
        let b = binder.require_tensor("w/b", NumericFormat::Bf16, StorageLayout::ContiguousLeV1, &[48, 64]).expect("b");
        assert!(binder.materialize_on_device_as_fp8(b, &[16, 16]).is_err(), "blocks short of the rows");
        let plan_before = binder.plan();
        assert_eq!(plan_before.device_objects.len(), 1, "a refused conversion places nothing");
        binder.materialize_on_device_as_fp8(b, &[16, 32]).expect("b converted");
        let plan = binder.finish().expect("all placed");
        let placed = plan.device_objects[1];
        assert_eq!(placed.encoding, DeviceEncoding::Fp8Rows(&[16, 32]));
        assert_eq!(placed.offset, 256);
        assert_eq!(placed.bytes, crate::fp8::block_offsets(&[16, 32], 64)[2]);
        assert_eq!(plan.device_capacity_bytes, 256 + placed.bytes);
        assert_eq!(plan.device_objects[0].encoding, DeviceEncoding::Stored);

        let mut binder = Binder::new(&reader);
        let a = binder.require_tensor("w/a", NumericFormat::Bf16, StorageLayout::ContiguousLeV1, &[4, 8]).expect("a");
        assert!(binder.materialize_on_device_as_fp8(a, &[4]).is_err(), "columns off 16");
        let (_fixture, reader) = fixture_reader("fp8-rows-nvfp4");
        let mut binder = Binder::new(&reader);
        let n = binder
            .require_tensor("w/nvfp4", NumericFormat::Nvfp4, StorageLayout::BlockScaleK16M128x4V1, &[128, 64])
            .expect("nvfp4");
        assert!(binder.materialize_on_device_as_fp8(n, &[128]).is_err(), "not BF16");
    }

    /// ADR 0002 happy path: every object consumed and placed -> `finish`
    /// succeeds and the plan covers all five objects.
    #[test]
    fn finish_succeeds_when_every_object_is_consumed_and_placed() {
        let (_artifact, reader) = fixture_reader("all-placed");
        let mut binder = Binder::new(&reader);
        bind_all(&mut binder);
        let plan = binder.finish().expect("every object consumed and placed");
        assert_eq!(plan.object_count, 5);
        assert_eq!(plan.device_objects.len(), 4);
        assert_eq!(plan.host_objects.len(), 1);
    }

    /// ADR 0002: an object left unconsumed is a load failure.
    #[test]
    fn finish_fails_when_an_object_is_unconsumed() {
        let (_artifact, reader) = fixture_reader("unconsumed");
        let mut binder = Binder::new(&reader);
        // Consume + place the four tensors, but never consume the resource.
        bind_devices(&mut binder);
        // The resource was never consumed -> finish must fail.
        let err = binder.finish().expect_err("finish must fail on an unconsumed object");
        assert!(err.to_string().contains("not consumed"), "{err}");
    }

    /// ADR 0002: a consumed object with no placement is a load failure.
    #[test]
    fn finish_fails_when_an_object_has_no_placement() {
        let (_artifact, reader) = fixture_reader("unplanned");
        let mut binder = Binder::new(&reader);
        // Consume every object, but leave the resource consumed-but-unplaced.
        let resource = binder
            .require_resource("frontend/tokenizer.json", ResourceEncoding::RawBytesV1)
            .expect("resource");
        bind_devices(&mut binder);
        // The resource is consumed but has no placement -> finish must fail.
        let _ = resource;
        let err = binder.finish().expect_err("finish must fail on an unplanned object");
        assert!(err.to_string().contains("no materialization placement"), "{err}");
    }

    /// A second bind of the same object is an error (consume-once contract).
    #[test]
    fn a_second_bind_of_the_same_object_is_rejected() {
        let (_artifact, reader) = fixture_reader("double-bind");
        let mut binder = Binder::new(&reader);
        binder
            .require_tensor("w/bf16", NumericFormat::Bf16, StorageLayout::ContiguousLeV1, &[4, 8])
            .expect("first bind");
        let err = binder
            .require_tensor("w/bf16", NumericFormat::Bf16, StorageLayout::ContiguousLeV1, &[4, 8])
            .expect_err("a second bind must fail");
        assert!(err.to_string().contains("more than once"), "{err}");
    }

    /// A second placement of the same object is an error.
    #[test]
    fn a_second_placement_of_the_same_object_is_rejected() {
        let (_artifact, reader) = fixture_reader("double-place");
        let mut binder = Binder::new(&reader);
        let bf16 = binder
            .require_tensor("w/bf16", NumericFormat::Bf16, StorageLayout::ContiguousLeV1, &[4, 8])
            .expect("bf16");
        binder.materialize_on_device(bf16).expect("first placement");
        let err = binder
            .materialize_on_device(bf16)
            .expect_err("a second placement must fail");
        assert!(
            err.to_string().contains("more than one materialization placement"),
            "{err}"
        );
    }

    /// The host-side placements: a tensor kept in host memory, one placed in
    /// the expert pool, one handed over as a file range. All count as placed
    /// for ADR 0002, and none of them takes device arena space.
    #[test]
    fn expert_pool_streamed_and_host_tensor_placements_complete_the_bind() {
        let (_artifact, reader) = fixture_reader("host-roles");
        let mut binder = Binder::new(&reader);
        let resource = binder
            .require_resource("frontend/tokenizer.json", ResourceEncoding::RawBytesV1)
            .expect("resource");
        binder.retain_on_host(resource).expect("retain resource");
        let bf16 = binder
            .require_tensor("w/bf16", NumericFormat::Bf16, StorageLayout::ContiguousLeV1, &[4, 8])
            .expect("bf16");
        binder.retain_tensor_on_host(bf16).expect("host tensor");
        let (nvfp4, format) = binder
            .require_tensor_of(
                "w/nvfp4",
                &[NumericFormat::Q4G64F16S, NumericFormat::Nvfp4],
                StorageLayout::BlockScaleK16M128x4V1,
                &[128, 64],
            )
            .expect("one of two formats");
        assert_eq!(format, NumericFormat::Nvfp4);
        assert_eq!(binder.place_in_expert_pool(nvfp4).expect("pool"), 0);
        let q4 = binder
            .require_tensor("w/q4", NumericFormat::Q4G64F16S, StorageLayout::RowSplitK128V1, &[128, 128])
            .expect("q4");
        // 4612 bytes of NVFP4, then the next pool slot at its alignment.
        assert_eq!(binder.place_in_expert_pool(q4).expect("pool"), 4864);
        let fp8 = binder
            .require_tensor("w/fp8", NumericFormat::Fp8E4M3FnRowBf16S, StorageLayout::RowScaleV1, &[128, 64])
            .expect("fp8");
        binder.stream_from_host(fp8).expect("streamed");

        let plan = binder.finish().expect("every object consumed and placed");
        assert_eq!(plan.device_capacity_bytes, 0);
        assert!(plan.device_objects.is_empty());
        assert_eq!(plan.host_objects.len(), 1, "the resource");
        assert_eq!(plan.host_tensor_objects, [HostPlacement { handle: bf16 }]);
        assert_eq!(plan.expert_pool_capacity_bytes, 4864 + 8704);
        let streamed = plan.streamed_objects[0];
        assert_eq!(streamed.handle, fp8);
        assert_eq!(streamed.bytes, 8448);
        assert_eq!(streamed.file_offset, reader.payload_offset() + 17920);
    }

    /// ADR 0062: a tensor in mapped pinned host memory takes no device arena
    /// space. Its placement is the aligned file span the materializer reads
    /// straight into the allocation, each span at a 4096-aligned offset, and
    /// the tensor sits as far into its span as its payload is past the span's
    /// start.
    #[test]
    fn a_host_mapped_placement_holds_the_aligned_file_span_and_no_device_space() {
        let (_artifact, reader) = fixture_reader("host-mapped");
        let mut binder = Binder::new(&reader);
        let bf16 = binder
            .require_tensor("w/bf16", NumericFormat::Bf16, StorageLayout::ContiguousLeV1, &[4, 8])
            .expect("bf16");
        binder.materialize_on_device(bf16).expect("device");
        let q4 = binder
            .require_tensor("w/q4", NumericFormat::Q4G64F16S, StorageLayout::RowSplitK128V1, &[128, 128])
            .expect("q4");
        binder.place_in_host_mapped(q4).expect("host mapped");
        let fp8 = binder
            .require_tensor("w/fp8", NumericFormat::Fp8E4M3FnRowBf16S, StorageLayout::RowScaleV1, &[128, 64])
            .expect("fp8");
        binder.place_in_host_mapped(fp8).expect("host mapped");

        let plan = binder.plan();
        assert_eq!(plan.device_capacity_bytes, 64, "only the BF16 tensor is on the device");
        assert_eq!(plan.device_objects.len(), 1);
        let payload = reader.payload_offset();
        // q4: payload bytes 9216..17920, read as the span 8192..20480.
        // fp8: payload bytes 17920..26368, read as the span 16384..28672,
        // placed after q4's span.
        assert_eq!(
            plan.host_mapped_objects,
            [
                HostMappedPlacement {
                    handle: q4,
                    span_offset: 0,
                    span_file_offset: payload + 8192,
                    span_bytes: 12288,
                    offset: 1024,
                    bytes: 8704,
                },
                HostMappedPlacement {
                    handle: fp8,
                    span_offset: 12288,
                    span_file_offset: payload + 16384,
                    span_bytes: 12288,
                    offset: 12288 + 1536,
                    bytes: 8448,
                },
            ]
        );
        assert_eq!(plan.host_mapped_capacity_bytes, 24576);
    }

    /// A resource is not a host tensor or a pool tensor, nothing is placed
    /// twice, and a format outside the family is refused.
    #[test]
    fn host_placements_refuse_the_wrong_kind_and_a_second_placement() {
        let (_artifact, reader) = fixture_reader("host-refusals");
        let mut binder = Binder::new(&reader);
        let resource = binder
            .require_resource("frontend/tokenizer.json", ResourceEncoding::RawBytesV1)
            .expect("resource");
        assert!(binder.retain_tensor_on_host(resource).is_err());
        assert!(binder.place_in_expert_pool(resource).is_err());
        assert!(binder.place_in_host_mapped(resource).is_err());
        let bf16 = binder
            .require_tensor("w/bf16", NumericFormat::Bf16, StorageLayout::ContiguousLeV1, &[4, 8])
            .expect("bf16");
        binder.stream_from_host(bf16).expect("streamed");
        let err = binder.place_in_expert_pool(bf16).expect_err("a second placement");
        assert!(err.to_string().contains("more than one materialization placement"), "{err}");
        let err = binder
            .require_tensor_of("w/q4", &[NumericFormat::Nvfp4], StorageLayout::RowSplitK128V1, &[128, 128])
            .expect_err("a format outside the family");
        assert!(err.to_string().contains("does not match target contract"), "{err}");
    }
}
