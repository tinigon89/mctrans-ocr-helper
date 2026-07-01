//! Manga colorization (Mode A, automatic) via the manga-colorization-v2
//! generator exported to ONNX and run with `ort` (ONNX Runtime).
//!
//! The generator takes a 5-channel input `[1, 5, H, W]`:
//!   ch0 = grayscale in [0,1]   (plain pixel/255, NOT normalized to [-1,1])
//!   ch1..4 = 0                  (color hint 3ch + mask 1ch — empty in auto mode)
//! and returns `[1, 3, H, W]` in ~[-1,1] (tanh); de-normalize with `out*0.5+0.5`.
//! H and W must be multiples of 32.

use anyhow::Result;
use image::{DynamicImage, RgbImage};
use ort::{session::Session, value::Tensor};

/// Load the colorizer ONNX into a session. Registers the CUDA execution
/// provider but falls back to CPU if it can't initialise, so the helper still
/// works without a CUDA-enabled ONNX Runtime.
pub fn load(path: &std::path::Path) -> Result<Session> {
    use ort::execution_providers::{CPUExecutionProvider, DirectMLExecutionProvider};
    // DirectML runs on any DX12 GPU (incl. NVIDIA) with no CUDA/cuDNN version
    // matching; falls through to CPU when the GPU/EP is unavailable. Setting
    // COLORIZE_EP=dml makes DirectML mandatory (hard error) — used by the test to
    // prove the GPU path; COLORIZE_EP=cpu forces CPU.
    let ep = std::env::var("COLORIZE_EP").unwrap_or_default();
    let providers = match ep.as_str() {
        "dml" => vec![DirectMLExecutionProvider::default().build().error_on_failure()],
        "cpu" => vec![CPUExecutionProvider::default().build()],
        _ => vec![
            DirectMLExecutionProvider::default().build(),
            CPUExecutionProvider::default().build(),
        ],
    };
    let session = Session::builder()?
        .with_execution_providers(providers)?
        .commit_from_file(path)?;
    Ok(session)
}

/// Colorize one tile. Scales to a model-friendly size (short side ~576, both
/// dims a multiple of 32), runs the generator, then upscales the RGB result back
/// to the tile's native resolution.
pub fn colorize_tile(session: &mut Session, tile: &DynamicImage) -> Result<RgbImage> {
    let (w0, h0) = (tile.width(), tile.height());
    let (w, h) = fit32(w0, h0, 576);
    let gray = image::imageops::resize(
        &tile.to_luma8(),
        w,
        h,
        image::imageops::FilterType::Triangle,
    );

    // NCHW [1,5,h,w]: ch0 = gray/255; ch1..4 = 0.
    let (hu, wu) = (h as usize, w as usize);
    let plane = hu * wu;
    let mut data = vec![0f32; 5 * plane];
    for (x, y, p) in gray.enumerate_pixels() {
        data[(y as usize) * wu + (x as usize)] = p[0] as f32 / 255.0;
    }

    let input = Tensor::from_array(([1usize, 5, hu, wu], data))?;
    let outputs = session.run(ort::inputs!["input" => input])?;
    let (_shape, out) = outputs["output"].try_extract_tensor::<f32>()?;

    // out = [1,3,h,w] in ~[-1,1]; de-normalize + clamp to bytes.
    let mut rgb = RgbImage::new(w, h);
    for y in 0..hu {
        for x in 0..wu {
            let idx = y * wu + x;
            let ch = |c: usize| {
                let v = out[c * plane + idx] * 0.5 + 0.5;
                (v.clamp(0.0, 1.0) * 255.0) as u8
            };
            rgb.put_pixel(x as u32, y as u32, image::Rgb([ch(0), ch(1), ch(2)]));
        }
    }

    Ok(image::imageops::resize(
        &rgb,
        w0,
        h0,
        image::imageops::FilterType::Triangle,
    ))
}

/// Scale so the short side ≈ `short`, both dims rounded up to a multiple of 32.
fn fit32(w: u32, h: u32, short: u32) -> (u32, u32) {
    let (mut nw, mut nh) = if h <= w {
        (((w as f32 * short as f32 / h as f32).round() as u32).max(32), short)
    } else {
        (short, ((h as f32 * short as f32 / w as f32).round() as u32).max(32))
    };
    nw += (32 - nw % 32) % 32;
    nh += (32 - nh % 32) % 32;
    (nw, nh)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Run: set COLORIZE_TEST_IMG to a B&W page and (optionally) COLORIZER_ONNX,
    // then `cargo test --features directml colorize_sample -- --nocapture`.
    // Skips (passes) when COLORIZE_TEST_IMG is unset so normal `cargo test`
    // doesn't need the 123MB model or a sample image.
    #[test]
    fn colorize_sample() {
        let Ok(img_path) = std::env::var("COLORIZE_TEST_IMG") else {
            eprintln!("COLORIZE_TEST_IMG unset — skipping colorize test");
            return;
        };
        let onnx = std::env::var("COLORIZER_ONNX").unwrap_or_else(|_| "models/colorizer.onnx".into());
        eprintln!("loading {onnx}");
        let mut sess = load(std::path::Path::new(&onnx)).expect("load onnx");
        let img = image::open(&img_path).expect("open image");
        let t = std::time::Instant::now();
        let out = colorize_tile(&mut sess, &img).expect("colorize");
        eprintln!(
            "colorized {}x{} -> {}x{} in {:?}",
            img.width(), img.height(), out.width(), out.height(), t.elapsed()
        );
        assert_eq!((out.width(), out.height()), (img.width(), img.height()));
        let out_path = "colorize_test_out.png";
        out.save(out_path).expect("save");
        eprintln!("wrote {out_path}");
    }
}
