//! Host-only 27B vision tower. Text inference and the finished embedding stay
//! on CUDA; tower weights, attention and projection intermediates do not.
//! This independent FP32 path is numerically close, not bit-identical, to CUDA.
use std::collections::HashMap;
use ignis_artifact::{normalize_tensor, NormalizedTensor, NumericFormat, Object, Reader};
use crate::vision::{vision_item_control, MediaItem, VisionItemControl};

const H: usize = 1152;
const HEADS: usize = 16;
const D: usize = 72;
const PATCH: usize = 1536;
const LAYERS: usize = 27;

struct Tensor { shape: Vec<usize>, data: Vec<f32> }

/// Decoded weights are owned for the load's lifetime, independent of its mmap.
pub struct CpuVision { weights: HashMap<String, Tensor>, threads: usize }

fn bf16(word: u16) -> f32 { f32::from_bits(u32::from(word) << 16) }
fn to_bf16(x: f32) -> u16 {
    if x.is_nan() { return ((x.to_bits() >> 16) as u16) | 0x40; }
    ((x.to_bits().wrapping_add(0x7fff + ((x.to_bits() >> 16) & 1))) >> 16) as u16
}
fn rounded(x: f32) -> f32 { bf16(to_bf16(x)) }

impl CpuVision {
    /// Host FP32 expansion budget, checked before any model allocation.
    pub fn planned_host_bytes() -> u64 {
        ignis_artifact::inventory::vision_scope_27b().iter()
            .map(|entry| entry.shape.iter().product::<u64>() * 4).sum()
    }

    pub fn load(reader: &Reader, threads: u32) -> Result<Self, String> {
        let mut weights = HashMap::new();
        for entry in ignis_artifact::inventory::vision_scope_27b() {
            let Some(Object::Tensor(t)) = reader.find(entry.name) else {
                return Err(format!("CPU vision missing {}", entry.name));
            };
            let payload = reader.payload(entry.name).map_err(|e| e.to_string())?;
            let normalized = normalize_tensor(&t.shape, t.format, t.layout, payload.data)
                .map_err(|e| format!("CPU vision {}: {e}", entry.name))?;
            let data = match normalized {
                NormalizedTensor::DequantBf16 { data, .. } => data.into_iter().map(bf16).collect(),
                NormalizedTensor::Contiguous { format: NumericFormat::Bf16, bytes, .. } =>
                    bytes.chunks_exact(2).map(|b| bf16(u16::from_le_bytes([b[0], b[1]]))).collect(),
                NormalizedTensor::Contiguous { format: NumericFormat::Fp32, bytes, .. } =>
                    bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
                _ => return Err(format!("CPU vision unsupported tensor format: {}", entry.name)),
            };
            weights.insert(entry.name.to_owned(), Tensor {
                shape: t.shape.iter().map(|&d| d as usize).collect(), data,
            });
        }
        let threads = if threads == 0 {
            std::thread::available_parallelism().map_or(1, usize::from)
        } else { threads as usize };
        Ok(Self { weights, threads })
    }

    pub fn host_bytes(&self) -> u64 {
        self.weights.values().map(|t| (t.data.len() * 4) as u64).sum()
    }

    fn tensor(&self, name: &str) -> &Tensor { &self.weights[name] }

    fn linear(&self, x: &[f32], name: &str, bias: &str) -> Vec<f32> {
        let w = self.tensor(name);
        let (out, input) = (w.shape[0], w.shape[1]);
        let mut y = product(x, &w.data, input, out, self.threads);
        let b = &self.tensor(bias).data;
        // CUDA's linear writes BF16 before the separate add_bias operation.
        for (i, v) in y.iter_mut().enumerate() { *v = rounded(rounded(*v) + b[i % out]); }
        y
    }

    fn norm(&self, x: &[f32], name: &str) -> Vec<f32> {
        norm(x, &self.tensor(&format!("{name}/weight")).data,
            &self.tensor(&format!("{name}/bias")).data)
    }

    pub fn encode(&self, item: &MediaItem) -> Result<Vec<u16>, String> {
        let grid = item.grid;
        if grid.t == 0 || grid.h == 0 || grid.w == 0 || grid.h % 2 != 0 || grid.w % 2 != 0 {
            return Err("CPU vision needs a positive merge-aligned grid".into());
        }
        let control = vision_item_control(grid);
        let patches = control.patches as usize;
        if item.patches.len() != patches * PATCH {
            return Err("CPU vision patch array does not match the grid".into());
        }
        let input: Vec<f32> = item.patches.iter().copied().map(bf16).collect();
        let mut x = self.linear(&input, "vision/patch_embedding", "vision/patch_embedding_bias");
        let table = &self.tensor("vision/position_embedding").data;
        for p in 0..patches {
            for d in 0..H {
                let mut pos = 0.0;
                for corner in 0..4 {
                    let index = control.position_table_indices[p * 4 + corner] as usize;
                    pos += table[index * H + d] * control.position_table_weights[p * 4 + corner];
                }
                x[p * H + d] = rounded(x[p * H + d] + pos);
            }
        }
        for layer in 0..LAYERS {
            let prefix = format!("vision/layers/{layer}");
            let normed = self.norm(&x, &format!("{prefix}/norm1"));
            let mut qkv = self.linear(&normed, &format!("{prefix}/attention/qkv"), &format!("{prefix}/attention/qkv_bias"));
            rotate(&mut qkv, &control);
            let attended = attention(&qkv, &control, self.threads);
            let projected = self.linear(&attended, &format!("{prefix}/attention/output"), &format!("{prefix}/attention/output_bias"));
            add_residual(&mut x, &projected);
            let normed = self.norm(&x, &format!("{prefix}/norm2"));
            let mut up = self.linear(&normed, &format!("{prefix}/mlp/fc1"), &format!("{prefix}/mlp/fc1_bias"));
            for v in &mut up { *v = rounded(0.5 * *v * (1.0 + (0.7978845608 * (*v + 0.044715 * *v * *v * *v)).tanh())); }
            let down = self.linear(&up, &format!("{prefix}/mlp/fc2"), &format!("{prefix}/mlp/fc2_bias"));
            add_residual(&mut x, &down);
        }
        // The processor already stores patches in adjacent 2x2 merge order.
        let merged = self.norm(&x, "vision/merger/norm");
        let mut hidden = self.linear(&merged, "vision/merger/fc1", "vision/merger/fc1_bias");
        for v in &mut hidden { *v = rounded(0.5 * *v * (1.0 + libm::erff(*v * std::f32::consts::FRAC_1_SQRT_2))); }
        let output = self.linear(&hidden, "vision/merger/fc2", "vision/merger/fc2_bias");
        Ok(output.into_iter().map(to_bf16).collect())
    }
}

fn product(x: &[f32], w: &[f32], input: usize, out: usize, threads: usize) -> Vec<f32> {
    assert_eq!(x.len() % input, 0);
    assert_eq!(w.len(), input * out);
    let rows = x.len() / input;
    let mut y = vec![0.0; rows * out];
    let block = rows.div_ceil(threads.max(1)).max(1);
    std::thread::scope(|scope| {
        for (chunk, y) in y.chunks_mut(block * out).enumerate() {
            let x = &x[chunk * block * input..][..y.len() / out * input];
            scope.spawn(move || unsafe {
                // All slices are checked above; workers own disjoint output rows.
                matrixmultiply::sgemm(y.len() / out, input, out, 1.0,
                    x.as_ptr(), input as isize, 1, w.as_ptr(), 1, input as isize,
                    0.0, y.as_mut_ptr(), out as isize, 1);
            });
        }
    });
    y
}

fn norm(x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
    let width = w.len();
    let mut y = Vec::with_capacity(x.len());
    for row in x.chunks_exact(width) {
        let mean = row.iter().sum::<f32>() / width as f32;
        let var = row.iter().map(|v| (*v - mean).powi(2)).sum::<f32>() / width as f32;
        let inv = (var + 1.0e-6).sqrt().recip();
        y.extend(row.iter().enumerate().map(|(i, v)| rounded((*v - mean) * inv * w[i] + b[i])));
    }
    y
}

fn add_residual(x: &mut [f32], y: &[f32]) {
    for (x, y) in x.iter_mut().zip(y) { *x = rounded(*x + *y); }
}

fn rotate(qkv: &mut [f32], c: &VisionItemControl) {
    let patches = c.patches as usize;
    for p in 0..patches {
        for side in 0..2 {
            for head in 0..HEADS {
                let base = p * 3 * H + side * H + head * D;
                for pair in 0..36 {
                    let axis = pair / 18;
                    let angle = c.position_ids[axis * patches + p] as f32
                        * 10000.0f32.powf(-2.0 * (pair % 18) as f32 / 36.0);
                    let (sin, cos) = angle.sin_cos();
                    let (a, b) = (qkv[base + pair], qkv[base + pair + 36]);
                    qkv[base + pair] = rounded(a * cos - b * sin);
                    qkv[base + pair + 36] = rounded(b * cos + a * sin);
                }
            }
        }
    }
}

fn attention(qkv: &[f32], c: &VisionItemControl, threads: usize) -> Vec<f32> {
    let patches = c.patches as usize;
    let mut output = vec![0.0; patches * H];
    // Assign independent query rows to workers; a worker holds only one
    // segment's scores, never an image-sized quadratic attention matrix.
    let block = patches.div_ceil(threads.max(1)).max(1);
    std::thread::scope(|scope| {
        for (chunk, dst) in output.chunks_mut(block * H).enumerate() {
            scope.spawn(move || {
                for (local, row) in dst.chunks_exact_mut(H).enumerate() {
                    let p = chunk * block + local;
                    let segment = c.cu_seqlens.windows(2).find(|s| p >= s[0] as usize && p < s[1] as usize).unwrap();
                    let (begin, end) = (segment[0] as usize, segment[1] as usize);
                    let mut scores = vec![0.0f32; end - begin];
                    for head in 0..HEADS {
                        let q = &qkv[p * 3 * H + head * D..][..D];
                        for (i, score) in scores.iter_mut().enumerate() {
                            let k = &qkv[(begin + i) * 3 * H + H + head * D..][..D];
                            *score = q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() / (D as f32).sqrt();
                        }
                        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let mut sum = 0.0;
                        for score in &mut scores { *score = (*score - max).exp(); sum += *score; }
                        for d in 0..D {
                            let v = scores.iter().enumerate().map(|(i, s)|
                                s / sum * qkv[(begin + i) * 3 * H + 2 * H + head * D + d]).sum();
                            row[head * D + d] = rounded(v);
                        }
                    }
                }
            });
        }
    });
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn product_uses_stored_row_major_weights_for_each_patch() {
        assert_eq!(product(&[1., 2., 3., 4.], &[1., 0., 0., 1., 2., 3.], 2, 3, 2),
            vec![1., 2., 8., 3., 4., 18.]);
    }
    #[test]
    fn bf16_rounds_ties_to_even_and_preserves_nan() {
        assert_eq!(to_bf16(f32::from_bits(0x3f808000)), 0x3f80);
        assert_eq!(to_bf16(f32::from_bits(0x3f818000)), 0x3f82);
        assert!(bf16(to_bf16(f32::NAN)).is_nan());
    }
    #[test]
    fn segmented_attention_cannot_read_another_frame() {
        let control = vision_item_control(ignis_artifact::vision::Grid { t: 2, h: 2, w: 2 });
        let mut qkv = vec![0.0; 8 * 3 * H];
        for p in 0..8 { qkv[p * 3 * H + 2 * H..p * 3 * H + 3 * H].fill(if p < 4 { 1.0 } else { 3.0 }); }
        let out = attention(&qkv, &control, 2);
        assert!(out[..4 * H].iter().all(|v| *v == 1.0));
        assert!(out[4 * H..].iter().all(|v| *v == 3.0));
    }
    #[test]
    fn zero_position_rotation_is_identity() {
        let mut c = vision_item_control(ignis_artifact::vision::Grid { t: 1, h: 2, w: 2 });
        c.position_ids.fill(0);
        let mut qkv = vec![1.0; 4 * 3 * H];
        rotate(&mut qkv, &c);
        assert!(qkv.iter().all(|v| *v == 1.0));
    }
}
