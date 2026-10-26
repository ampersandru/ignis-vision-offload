//! Times one step of the diarizer's forward pass by part (spec speech/08's "As built" times):
//! the q/k/v and output projections, the RoPE, the attention, the MLP and the head, over the rows
//! of a full step of a geometry (speaker cache, FIFO, chunk and look-ahead).
//!
//! usage: diarizer_profile MODEL.gguf [--geometry live|file] [--threads N] [--reps N]

use std::time::Instant;

use ignis_speech::cpu::{select, Features};
use ignis_speech::diarizer::{model, Geometry, Weights};
use ignis_speech::matmul::matmul;
use ignis_speech::ops::{self, AttnShape, Kv, Mask, Rope};
use ignis_speech::pool::Pool;
use ignis_speech::Threads;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let take = |args: &mut Vec<String>, name: &str| args.iter().position(|a| a == name).map(|i| { args.remove(i); args.remove(i) });
    let threads = take(&mut args, "--threads").map(|v| v.parse().unwrap()).unwrap_or(Threads::Auto.resolve().0);
    let reps: usize = take(&mut args, "--reps").map(|v| v.parse().unwrap()).unwrap_or(5);
    let g = match take(&mut args, "--geometry").as_deref() {
        Some("file") => Geometry::FILE,
        _ => Geometry::LIVE,
    };
    let w = Weights::load(std::path::Path::new(&args[0]), false).unwrap();
    let isa = select(Features::detect()).expect("AVX2");
    let pool = Pool::new(threads);
    let n = g.speaker_cache + g.fifo + g.chunk + g.right_context;
    let d = w.d;
    let hd = d / w.heads;
    let x: Vec<f32> = (0..n * d).map(|i| ((i * 2654435761usize) % 1000) as f32 / 1000.0 - 0.5).collect();
    let l = &w.layers[0];
    let ff = l.fc1.w.n;
    let rope = Rope::text(0, n, [hd / 2, 0, 0], w.theta, hd);
    let shape = AttnShape { hq: w.heads, hkv: w.heads, hd };
    let (mut q, mut k, mut v, mut att, mut o) = (vec![0f32; n * d], vec![0f32; n * d], vec![0f32; n * d], vec![0f32; n * d], vec![0f32; n * d]);
    let (mut f, mut y) = (vec![0f32; n * ff], vec![0f32; n * d]);
    let mut t = std::collections::BTreeMap::<&str, Vec<f64>>::new();
    for _ in 0..reps + 1 {
        let mut time = |name: &'static str, body: &mut dyn FnMut()| {
            let s = Instant::now();
            body();
            t.entry(name).or_default().push(s.elapsed().as_secs_f64() * 1e3);
        };
        time("qkv", &mut || {
            matmul(&pool, isa, &l.q, &x, n, &mut q);
            matmul(&pool, isa, &l.k, &x, n, &mut k);
            matmul(&pool, isa, &l.v, &x, n, &mut v);
        });
        time("rope", &mut || {
            rope.apply(isa, &mut q, w.heads);
            rope.apply(isa, &mut k, w.heads);
        });
        time("attention", &mut || ops::attention(&pool, isa, shape, &q, n, Kv::F32 { k: &k, v: &v }, n, Mask::Full, &mut att));
        time("out", &mut || matmul(&pool, isa, &l.o.w, &att, n, &mut o));
        time("fc1", &mut || matmul(&pool, isa, &l.fc1.w, &x, n, &mut f));
        time("gelu", &mut || ops::gelu(isa, &mut f));
        time("fc2", &mut || matmul(&pool, isa, &l.fc2.w, &f, n, &mut y));
        time("forward", &mut || drop(w.forward(&pool, isa, &x, n, n, None)));
        time("step attention", &mut || model::attention(&pool, isa, w.heads, &q, &k, &v, n, n, &mut att));
        time("gelu on the pool", &mut || model::rows(&pool, &mut f, ff, &|r| ops::gelu(isa, r)));
        time("layer_norm", &mut || {
            let mut a = x.clone();
            ops::layer_norm(isa, &mut a, &l.ln1.0, &l.ln1.1, w.eps);
        });
    }
    let layers = w.layers.len() as f64;
    let mut total = 0.0;
    println!("{n} rows, {threads} threads, {:?} weights, ms per layer (median of {reps}) and per step:", l.q.ty);
    for (name, v) in &t {
        let m = median(v[1..].to_vec());
        if ["forward", "step attention", "gelu on the pool"].contains(name) {
            println!("  {name:>10}: {m:7.1} ms per step (the whole graph)");
            continue;
        }
        let per = if *name == "layer_norm" { 2.0 * m } else { m };
        total += per * layers;
        println!("  {name:>10}: {m:7.2} ms  x{layers} = {:7.1} ms", per * layers);
    }
    let gflop = 2.0 * n as f64 * layers * (4.0 * (d * d) as f64 + 2.0 * (d * ff) as f64 + 2.0 * (n * d) as f64) / 1e9;
    println!("  encoder: {total:.0} ms, {gflop:.0} GFLOP, {:.0} GFLOP/s", gflop / total * 1e3);
}
