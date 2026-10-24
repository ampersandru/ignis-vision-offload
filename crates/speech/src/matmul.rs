//! Matmul weights and the threaded driver: a GEMV for one activation row, a
//! cache-blocked, register-tiled GEMM for more (spec speech/01 §1, the spike).
//!
//! `out (m × n, row-major) = x (m × k) · wᵀ`, where `w` holds `n` rows of
//! `k`. Each call quantizes its own input to the weight's activation format
//! (Q8_K for Q4_K and Q6_K, Q8_0 for Q8_0, none for F16 and F32).

use std::cell::RefCell;

use crate::cpu::Isa;
use crate::mem::{AlignedBuf, Blob};
use crate::pool::Pool;
use crate::quant::*;

/// A matmul weight: `n` rows of `k`, in format `ty`.
pub struct Weight {
    pub name: String,
    pub ty: WType,
    pub n: usize,
    pub k: usize,
    data: Blob,
}

impl Weight {
    /// A weight over `data` (the file's bytes, or a repacked buffer for Q4Kx8).
    pub fn new(name: &str, ty: WType, n: usize, k: usize, data: Blob) -> Weight {
        assert!(k % ty.k_multiple() == 0, "{name}: k {k} is not a whole number of blocks");
        assert!(ty != WType::Q4Kx8 || n % 8 == 0, "{name}: the interleaved Q4_K needs rows in groups of 8");
        assert_eq!(data.len(), ty.row_bytes(k) * n, "{name}: {} bytes for {n} rows of {k}", data.len());
        let align = if matches!(ty, WType::F32) { 4 } else { 2 };
        assert_eq!(data.as_ptr() as usize % align, 0, "{name}: data misaligned");
        Weight { name: name.into(), ty, n, k, data }
    }

    /// Repacks a Q4_K weight (the file's layout) into the interleaved one: the only copy kept.
    pub fn repack_q4k(name: &str, n: usize, k: usize, file: &[u8]) -> Weight {
        assert!(n % 8 == 0 && k % QK_K == 0, "{name}: {n} rows of {k} cannot be interleaved");
        let nb = k / QK_K;
        let rb = WType::Q4K.row_bytes(k);
        assert_eq!(file.len(), rb * n, "{name}: {} bytes for {n} rows of {k}", file.len());
        let mut buf = AlignedBuf::zeroed(n / 8 * nb * 1184);
        let dst = cast_slice_mut::<BlockQ4Kx8>(buf.as_mut_slice(), n / 8 * nb);
        for g in 0..n / 8 {
            let rows: [&[BlockQ4K]; 8] = std::array::from_fn(|c| cast_slice::<BlockQ4K>(&file[(8 * g + c) * rb..], nb));
            repack_q4k_x8(rows, &mut dst[g * nb..(g + 1) * nb]);
        }
        Weight { name: name.into(), ty: WType::Q4Kx8, n, k, data: Blob::owned(buf) }
    }

    pub fn bytes(&self) -> &[u8] {
        self.data.as_slice()
    }

    fn ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }
}

/// Register tiles: (activation rows, weight rows) for the GEMV (one activation row) and the GEMM.
/// For the interleaved Q4_K the second number counts groups of 8 weight rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tiles {
    pub gemv: (usize, usize),
    pub gemm: (usize, usize),
}

impl Tiles {
    /// The spike's tuned tiles (ADR 0067): interleaved Q4_K GEMV of 16 rows and GEMM 4 × 16,
    /// Q6_K GEMM 8 × 1, Q8_0 GEMM 6 × 2.
    pub fn default_for(ty: WType) -> Tiles {
        match ty {
            WType::Q4Kx8 => Tiles { gemv: (1, 2), gemm: (4, 2) },
            WType::Q4K => Tiles { gemv: (1, 1), gemm: (8, 1) },
            WType::Q6K => Tiles { gemv: (1, 1), gemm: (8, 1) },
            WType::Q80 => Tiles { gemv: (1, 2), gemm: (6, 2) },
            WType::F16 | WType::F32 => Tiles { gemv: (1, 4), gemm: (4, 3) },
        }
    }
}

/// GEMV tasks per thread, claimed dynamically.
const GEMV_TASKS_PER_THREAD: usize = 4;
/// Bytes of weight rows a GEMM task keeps in L2 while every activation block passes over them.
const W_TILE_BYTES: usize = 64 << 10;
/// Bytes of activation rows per block (kept in L2 while the weight tile passes).
const A_TILE_BYTES: usize = 96 << 10;

#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);
// SAFETY: the driver hands each task disjoint output elements and read-only inputs.
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    fn get(self) -> *mut T {
        self.0
    }
}

thread_local! {
    /// The quantized activation scratch of the thread that drives the pool.
    static ACT: RefCell<AlignedBuf> = RefCell::new(AlignedBuf::zeroed(0));
}

/// Quantizes `m` rows of `k` f32 into `out` in format `ty` (rows of `ty.row_bytes(k)`).
pub fn quantize(pool: &Pool, isa: Isa, ty: AType, x: &[f32], m: usize, k: usize, out: &mut [u8]) {
    let rb = ty.row_bytes(k);
    assert!(x.len() >= m * k && out.len() >= m * rb);
    let base = SendPtr(out.as_mut_ptr());
    let row = |r: usize| {
        // SAFETY: row r of the output is rb bytes inside `out`, written by this call alone.
        let dst = unsafe { std::slice::from_raw_parts_mut(base.get().add(r * rb), rb) };
        quantize_row(isa, ty, &x[r * k..(r + 1) * k], dst);
    };
    if m == 1 {
        row(0);
        return;
    }
    const ROWS: usize = 4;
    pool.run(m.div_ceil(ROWS), &|t| (t * ROWS..((t + 1) * ROWS).min(m)).for_each(row));
}

fn quantize_row(isa: Isa, ty: AType, x: &[f32], dst: &mut [u8]) {
    let k = x.len();
    #[cfg(target_arch = "x86_64")]
    if isa.avx2() {
        // SAFETY: an AVX2 Isa exists only after detection; x holds k floats and dst a row of ty.
        unsafe {
            match ty {
                AType::Q8K => crate::avx2::quantize_q8k(x.as_ptr(), dst.as_mut_ptr() as *mut BlockQ8K, k / QK_K),
                AType::Q80 => crate::avx2::quantize_q80(x.as_ptr(), dst.as_mut_ptr() as *mut BlockQ80, k / QK8_0),
                AType::F32 => unreachable!("f32 activations are not quantized"),
            }
        }
        return;
    }
    let _ = isa;
    match ty {
        AType::Q8K => quantize_q8k(x, cast_slice_mut(dst, k / QK_K)),
        AType::Q80 => quantize_q80(x, cast_slice_mut(dst, k / QK8_0)),
        AType::F32 => unreachable!("f32 activations are not quantized"),
    }
}

/// `out[r * n + c] = Σ_k x[r][k] · w[c][k]` with the weight's default tiles.
pub fn matmul(pool: &Pool, isa: Isa, w: &Weight, x: &[f32], m: usize, out: &mut [f32]) {
    matmul_tiled(pool, isa, w, x, m, out, Tiles::default_for(w.ty));
}

pub fn matmul_tiled(pool: &Pool, isa: Isa, w: &Weight, x: &[f32], m: usize, out: &mut [f32], tiles: Tiles) {
    let (n, k) = (w.n, w.k);
    assert!(x.len() >= m * k, "{}: {} inputs for {m} rows of {k}", w.name, x.len());
    assert!(out.len() >= m * n, "{}: {} outputs for {m} rows of {n}", w.name, out.len());
    if m == 0 {
        return;
    }
    let at = w.ty.act();
    if at == AType::F32 {
        run(pool, isa, w, x.as_ptr() as *const u8, k * 4, m, out, tiles);
        return;
    }
    let arb = at.row_bytes(k);
    ACT.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < m * arb {
            *buf = AlignedBuf::zeroed((m * arb).next_power_of_two());
        }
        quantize(pool, isa, at, x, m, k, buf.as_mut_slice());
        run(pool, isa, w, buf.as_ptr(), arb, m, out, tiles);
    });
}

/// The quantized (or f32) activation rows `a` at stride `arb` against the weight.
#[allow(clippy::too_many_arguments)]
fn run(pool: &Pool, isa: Isa, w: &Weight, a: *const u8, arb: usize, m: usize, out: &mut [f32], tiles: Tiles) {
    let ap = SendPtr(a as *mut u8);
    let op = SendPtr(out.as_mut_ptr());
    #[cfg(target_arch = "x86_64")]
    if isa.avx2() {
        if w.ty == WType::Q4Kx8 {
            run_x8_avx2(pool, w, ap, arb, m, op, tiles);
        } else {
            run_avx2(pool, w, ap, arb, m, op, tiles);
        }
        return;
    }
    let _ = (isa, tiles);
    run_scalar(pool, w, ap, arb, m, op);
}

fn run_scalar(pool: &Pool, w: &Weight, ap: SendPtr<u8>, arb: usize, m: usize, op: SendPtr<f32>) {
    let (n, k, ty) = (w.n, w.k, w.ty);
    let data = w.bytes();
    const ROWS: usize = 16;
    pool.run(n.div_ceil(ROWS), &|t| {
        for c in t * ROWS..((t + 1) * ROWS).min(n) {
            for r in 0..m {
                // SAFETY: activation row r lies inside the caller's buffer of m rows at stride arb.
                let a = unsafe { std::slice::from_raw_parts(ap.get().add(r * arb) as *const u8, arb) };
                let y = dot_scalar(ty, data, c, k, a);
                // SAFETY: element (r, c) is written by this task alone.
                unsafe { *op.get().add(r * n + c) = y };
            }
        }
    });
}

/// Row `c` of the weight against one activation row, by the scalar reference.
pub fn dot_scalar(ty: WType, data: &[u8], c: usize, k: usize, a: &[u8]) -> f32 {
    let rb = ty.row_bytes(k);
    match ty {
        WType::Q4K => dot_q4k_q8k(cast_slice(&data[c * rb..], k / QK_K), cast_slice(a, k / QK_K)),
        WType::Q4Kx8 => {
            let nb = k / QK_K;
            dot_q4kx8_q8k(cast_slice(&data[c / 8 * nb * 1184..], nb), c % 8, cast_slice(a, nb))
        }
        WType::Q6K => dot_q6k_q8k(cast_slice(&data[c * rb..], k / QK_K), cast_slice(a, k / QK_K)),
        WType::Q80 => dot_q80_q80(cast_slice(&data[c * rb..], k / QK8_0), cast_slice(a, k / QK8_0)),
        WType::F16 => dot_f16_f32(cast_slice(&data[c * rb..], k), cast_slice(a, k)),
        WType::F32 => dot_f32_f32(cast_slice(&data[c * rb..], k), cast_slice(a, k)),
    }
}

#[cfg(target_arch = "x86_64")]
fn run_avx2(pool: &Pool, w: &Weight, ap: SendPtr<u8>, arb: usize, m: usize, op: SendPtr<f32>, tiles: Tiles) {
    use crate::avx2::tile_fn;
    let (n, k, ty) = (w.n, w.k, w.ty);
    let wrb = ty.row_bytes(k);
    let nb = match ty {
        WType::F16 | WType::F32 => k,
        _ => k / ty.k_multiple(),
    };
    let wp = SendPtr(w.ptr() as *mut u8);
    let nt = pool.threads();
    let tile = |mr, nr| tile_fn(ty, mr, nr).unwrap_or_else(|| panic!("no {ty:?} tile {mr}×{nr}"));

    if m == 1 {
        let nr = tiles.gemv.1;
        let (f, f1) = (tile(1, nr), tile(1, 1));
        let chunk = n.div_ceil(nt * GEMV_TASKS_PER_THREAD).next_multiple_of(8.max(nr));
        pool.run(n.div_ceil(chunk), &|t| {
            let (c1, mut c) = (((t + 1) * chunk).min(n), t * chunk);
            // SAFETY: rows c..c1 of the weight and output columns c..c1 are this task's.
            unsafe {
                while c + nr <= c1 {
                    f(wp.get().add(c * wrb), wrb, ap.get(), arb, nb, op.get().add(c), n);
                    c += nr;
                }
                while c < c1 {
                    f1(wp.get().add(c * wrb), wrb, ap.get(), arb, nb, op.get().add(c), n);
                    c += 1;
                }
            }
        });
        return;
    }

    let (mr, nr) = tiles.gemm;
    let (fmn, fm1, f1n, f11) = (tile(mr, nr), tile(mr, 1), tile(1, nr), tile(1, 1));
    // A weight tile of about W_TILE_BYTES, a multiple of nr, and enough tasks to balance.
    let wt = ((W_TILE_BYTES / wrb).max(nr) / nr * nr).min(n.div_ceil(nt * 4).next_multiple_of(nr)).max(nr);
    let mb = ((A_TILE_BYTES / arb).max(mr) / mr * mr).max(mr);
    pool.run(n.div_ceil(wt), &|t| {
        let (c0, c1) = (t * wt, ((t + 1) * wt).min(n));
        let mut m0 = 0;
        // SAFETY: weight rows c0..c1 and output columns c0..c1 (all m rows) are this task's.
        unsafe {
            while m0 < m {
                let m1 = (m0 + mb).min(m);
                let mut c = c0;
                while c < c1 {
                    let (fa, fb, cn) = if c + nr <= c1 { (fmn, f1n, nr) } else { (fm1, f11, 1) };
                    let wpc = wp.get().add(c * wrb);
                    let mut r = m0;
                    while r + mr <= m1 {
                        fa(wpc, wrb, ap.get().add(r * arb), arb, nb, op.get().add(r * n + c), n);
                        r += mr;
                    }
                    while r < m1 {
                        fb(wpc, wrb, ap.get().add(r * arb), arb, nb, op.get().add(r * n + c), n);
                        r += 1;
                    }
                    c += cn;
                }
                m0 = m1;
            }
        }
    });
}

#[cfg(target_arch = "x86_64")]
fn run_x8_avx2(pool: &Pool, w: &Weight, ap: SendPtr<u8>, arb: usize, m: usize, op: SendPtr<f32>, tiles: Tiles) {
    use crate::avx2::tile_x8_fn;
    let (n, k) = (w.n, w.k);
    let nb = k / QK_K;
    let gbytes = nb * 1184;
    let ng = n / 8;
    let wp = SendPtr(w.ptr() as *mut u8);
    let nt = pool.threads();
    let tile = |mr, g| tile_x8_fn(mr, g).unwrap_or_else(|| panic!("no interleaved tile {mr}×{g}"));

    if m == 1 {
        let gv = tiles.gemv.1;
        let (f, f1) = (tile(1, gv), tile(1, 1));
        let chunk = ng.div_ceil(nt * GEMV_TASKS_PER_THREAD).next_multiple_of(gv).max(gv);
        pool.run(ng.div_ceil(chunk), &|t| {
            let (g1, mut g) = (((t + 1) * chunk).min(ng), t * chunk);
            // SAFETY: groups g..g1 and their 8·(g1−g) output columns are this task's.
            unsafe {
                while g + gv <= g1 {
                    f(wp.get().add(g * gbytes), gbytes, ap.get(), arb, nb, op.get().add(8 * g), n);
                    g += gv;
                }
                while g < g1 {
                    f1(wp.get().add(g * gbytes), gbytes, ap.get(), arb, nb, op.get().add(8 * g), n);
                    g += 1;
                }
            }
        });
        return;
    }

    let (mr, gg) = tiles.gemm;
    let (fmn, fm1, f1n, f11) = (tile(mr, gg), tile(mr, 1), tile(1, gg), tile(1, 1));
    let wt = ((W_TILE_BYTES / gbytes).max(gg) / gg * gg).min(ng.div_ceil(nt * 4).next_multiple_of(gg)).max(gg);
    let mb = ((A_TILE_BYTES / arb).max(mr) / mr * mr).max(mr);
    pool.run(ng.div_ceil(wt), &|t| {
        let (g0, g1) = (t * wt, ((t + 1) * wt).min(ng));
        let mut m0 = 0;
        // SAFETY: groups g0..g1 and their output columns (all m rows) are this task's.
        unsafe {
            while m0 < m {
                let m1 = (m0 + mb).min(m);
                let mut g = g0;
                while g < g1 {
                    let (fa, fb, gn) = if g + gg <= g1 { (fmn, f1n, gg) } else { (fm1, f11, 1) };
                    let wpg = wp.get().add(g * gbytes);
                    let mut r = m0;
                    while r + mr <= m1 {
                        fa(wpg, gbytes, ap.get().add(r * arb), arb, nb, op.get().add(r * n + 8 * g), n);
                        r += mr;
                    }
                    while r < m1 {
                        fb(wpg, gbytes, ap.get().add(r * arb), arb, nb, op.get().add(r * n + 8 * g), n);
                        r += 1;
                    }
                    g += gn;
                }
                m0 = m1;
            }
        }
    });
}

#[cfg(test)]
#[path = "matmul_tests.rs"]
mod tests;
