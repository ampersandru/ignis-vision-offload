//! Times the executor's units on a real GGUF pair: the convolutional stem of one 8 s window,
//! a whole window's encode, the decoder prefill of its 104 tokens, and decode steps; the
//! breakdown behind spec speech/01's "As built" times.
//!
//! usage: profile DECODER.gguf ENCODER.gguf [--threads N] [--reps N]

use std::time::Instant;

use ignis_speech::cpu::{select, Features};
use ignis_speech::decoder::{self, Cache, Logits};
use ignis_speech::encoder::encode_window;
use ignis_speech::matmul::matmul;
use ignis_speech::model::Model;
use ignis_speech::ops;
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
    let isa = select(Features::detect()).expect("AVX2");
    let pool = Pool::new(threads);
    let t = Instant::now();
    let m = Model::load(args[0].as_ref(), args[1].as_ref(), false).unwrap();
    println!("load {:.0} ms, threads {threads}", t.elapsed().as_secs_f64() * 1e3);
    let e = &m.encoder;
    let mel: Vec<f32> = (0..800 * 128).map(|i| ((i * 7919) % 1000) as f32 / 1000.0 - 0.5).collect();
    let (mut stem, mut enc, mut pre, mut dec) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..reps {
        let t = Instant::now();
        let (mut a, mut b) = (Vec::new(), Vec::new());
        let mut rows = Vec::new();
        for _ in 0..8 {
            let plane = vec![0.1f32; 128 * 100];
            ops::conv3x3s2_gelu(&pool, isa, &e.conv[0].w, e.conv[0].b.as_deref().unwrap(), &plane, 1, 128, 100, &mut a);
            ops::conv3x3s2_gelu(&pool, isa, &e.conv[1].w, e.conv[1].b.as_deref().unwrap(), &a, 480, 64, 50, &mut b);
            ops::conv3x3s2_gelu(&pool, isa, &e.conv[2].w, e.conv[2].b.as_deref().unwrap(), &b, 480, 32, 25, &mut a);
            rows.extend_from_slice(&a[..13 * 7680]);
        }
        let mut h = vec![0f32; 104 * e.d];
        matmul(&pool, isa, &e.conv_out, &rows, 104, &mut h);
        stem.push(t.elapsed().as_secs_f64());

        let t = Instant::now();
        let x = encode_window(e, &pool, isa, &mel, 800, None);
        enc.push(t.elapsed().as_secs_f64());

        let mut cache = Cache::new(&m.decoder, 1024);
        let head = decoder::embed(&m.decoder, &vec![151644u32; 200]);
        decoder::forward(&m.decoder, &pool, isa, &mut cache, head, Logits::None, None);
        let t = Instant::now();
        decoder::forward(&m.decoder, &pool, isa, &mut cache, x, Logits::Last, None);
        pre.push(t.elapsed().as_secs_f64());

        let t = Instant::now();
        let mut tok = 11528u32;
        for _ in 0..20 {
            let l = decoder::forward(&m.decoder, &pool, isa, &mut cache, decoder::embed(&m.decoder, &[tok]), Logits::Last, None);
            tok = decoder::argmax(&l);
        }
        dec.push(t.elapsed().as_secs_f64() / 20.0);
    }
    println!(
        "window stem {:.1} ms, window encode {:.1} ms, prefill 104 tokens at 200 {:.1} ms, decode {:.2} ms/token at ~310",
        median(stem) * 1e3,
        median(enc) * 1e3,
        median(pre) * 1e3,
        median(dec) * 1e3
    );
}
