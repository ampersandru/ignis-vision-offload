//! Byte storage for weights: owned and 64-byte aligned (the repacked Q4_K,
//! scratch), or a view into a mapped file or a pinned copy kept alive by its
//! owner.

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::sync::Arc;

/// An owned, zeroed, 64-byte aligned buffer.
pub struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: the buffer is plain bytes owned by this value.
unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    fn layout(len: usize) -> Layout {
        Layout::from_size_align(len.max(64), 64).expect("buffer layout")
    }

    pub fn zeroed(len: usize) -> AlignedBuf {
        // SAFETY: the layout is non-zero sized.
        let ptr = unsafe { alloc_zeroed(Self::layout(len)) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(Self::layout(len));
        }
        AlignedBuf { ptr, len }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is valid for len initialised bytes.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, and the borrow is exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: allocated in `zeroed` with this layout.
        unsafe { dealloc(self.ptr, Self::layout(self.len)) }
    }
}

/// Read-only bytes: owned, or borrowed from storage `owner` keeps alive. Only the crate
/// borrows, since nothing outside it can tie a view to its owner:
///
/// ```compile_fail
/// let _ = ignis_speech::mem::Blob::borrowed(&[0u8; 4], std::sync::Arc::new(()));
/// ```
pub struct Blob {
    ptr: *const u8,
    len: usize,
    _owner: Arc<dyn Send + Sync>,
}

// SAFETY: the bytes are immutable for the blob's life and the owner is Send + Sync.
unsafe impl Send for Blob {}
unsafe impl Sync for Blob {}

impl Blob {
    pub fn owned(buf: AlignedBuf) -> Blob {
        let (ptr, len) = (buf.as_ptr(), buf.len());
        Blob { ptr, len, _owner: Arc::new(buf) }
    }

    /// A view of `bytes`, which must live inside storage `owner` keeps alive and never mutates.
    /// Nothing in the signature ties the two, so it is the crate's own: the model's tensors,
    /// borrowed from the mapped file or the pinned copy that `owner` is.
    pub(crate) fn borrowed(bytes: &[u8], owner: Arc<dyn Send + Sync>) -> Blob {
        Blob { ptr: bytes.as_ptr(), len: bytes.len(), _owner: owner }
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: the owner keeps ptr valid for len bytes for the blob's life.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}
