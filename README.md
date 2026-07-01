# mctrans-ocr-helper

Local OCR helper for MC-Trans comic translation. A thin HTTP server wrapping
[koharu-ml](https://github.com/mayocream/koharu)'s Candle models:

- **Detect** text regions — `ComicTextDetector`
- **Recognise** each region — **PaddleOCR-VL** (manga-capable VLM)
- **Inpaint** (optional) — **LaMa** removes original text → clean background
- **Colorize** (optional) — automatic B&W → colour via manga-colorization-v2
  (ONNX Runtime + **DirectML** GPU); see [Manga colorization](#manga-colorization-directml)

Runs on the user's own machine (GPU via CUDA/Metal, CPU fallback). The web app
and browser extension call it over loopback, so OCR runs natively instead of in
WASM. ~80 MB binary; model weights download from Hugging Face on first run.

> **License:** GPL-3.0 (links koharu-ml). Keep this in its **own repo** — do
> not vendor it into the proprietary MC-Trans web repo. The web app only talks
> to it over HTTP (separate process), so the web app stays proprietary.

## API

```
GET  /health                         -> { ok, name, version, device }
POST /ocr-page   (multipart)         -> { boxes: OcrBox[], cleanedImage? }
        file=<image bytes>           (required)
        inpaint=true                 (optional → also returns cleanedImage data URL)

# Staged (single-page editor)
POST /detect         (file)          -> { boxes, mask }         (boxes + segmentation mask)
POST /ocr            (file, boxes)   -> { boxes }               (OCR the given boxes)
POST /inpaint        (file)          -> { cleanedImage }        (whole-page text removal)
POST /inpaint-region (file, boxes)   -> { cleanedImage }        (erase only the given boxes)
POST /colorize       (file[, size])  -> { colorizedImage }      (AI colorize; size = model
                                                                 short-side px, default 768)
```

`OcrBox` matches MC-Trans's `OCRBox` shape (fractional coords):

```jsonc
{ "id": "…", "text": "…",
  "x": 0.12, "y": 0.34, "width": 0.20, "height": 0.08,
  "confidence": 0.97, "rotation": 0 }
```

## Build & run

### CPU
```bash
cargo run --release
```

### NVIDIA GPU (CUDA) — full-GPU pipeline, ~2.4 s/page on an RTX 4090
Both ML backends run on the GPU: candle (detection + inpaint) and llama.cpp
(GGUF OCR). Neither bundles CUDA — at first launch the runtime is downloaded to
match the hardware (CUDA 13.1 for NVIDIA, Vulkan otherwise), exactly like koharu.

**Build against CUDA 13.x** so candle's `cublas64_13.dll` lookup matches that
auto-downloaded runtime (build with 12.x and it would look for `cublas64_12`,
which isn't in the download → would need the toolkit on PATH). Build env needs:
cmake, LLVM/libclang, **CUDA 13.x toolkit (nvcc)**, and **MSVC cl.exe** (nvcc's
host compiler):

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_PATH=H:\NVIDIA\CUDA13.1"
set "PATH=%CUDA_PATH%\bin;C:\Program Files\CMake\bin;C:\Program Files\LLVM\bin;%PATH%"
set "LIBCLANG_PATH=C:\Program Files\LLVM\bin"
cargo build --release --features cuda
```

Set `CUDA_COMPUTE_CAP` in `.cargo/config.toml` to your GPU (89 = RTX 4090 / Ada).
If you switch CUDA toolkit versions, `cargo clean -p cudarc -p candle-kernels`
first so they recompile against the new one.

The built binary is **self-contained**: it runs with nothing CUDA on PATH (the
toolkit is only needed to *build*). Just run the exe — or `run.bat`.

> Gotchas baked into the code/config:
> - **`[patch.crates-io]`** in `Cargo.toml` mirrors koharu's forked candle + ug
>   (cudarc `dynamic-loading`); without it the cuda build fails with "both
>   dynamic-loading and dynamic-linking active".
> - `runtime.prepare()` runs **before** the candle models load, so the downloaded
>   CUDA 13.1 DLLs (`cublas64_13` …) are preloaded + on the DLL search path when
>   candle's cudarc resolves them.
> - The entry + tokio threads use a 64 MB stack (cuDNN's DllMain is stack-hungry;
>   the default ~1 MB Windows stack overflows while loading it).

### Apple Silicon
```bash
cargo run --release --features metal
```

Serves on `http://127.0.0.1:7842`. Override the model cache dir with
`MCTRANS_HELPER_ROOT`.

Smoke test:

```bash
curl -s http://127.0.0.1:7842/health
curl -s -F file=@page.png -F inpaint=true http://127.0.0.1:7842/ocr-page | jq '.boxes | length'
```

## Distribution

`cargo build --release --features cuda` → a single **~80 MB** binary. Ship just
the exe via GitHub Releases — **no CUDA DLLs to bundle**. On first launch it
downloads, into the model-cache dir:
- the matching prebuilt runtime (CUDA 13.1 / Vulkan / CPU) for the user's hardware,
- llama.cpp, and the model weights.

End users need **no Python, no CUDA toolkit** — only an NVIDIA driver for the GPU
path (AMD/Intel get Vulkan OCR + CPU detection; no GPU → all CPU).

## Manga colorization (DirectML)

`POST /colorize` runs **manga-colorization-v2**'s generator (exported to ONNX)
via **ONNX Runtime** with the **DirectML** execution provider — GPU on any DX12
card (incl. NVIDIA) with **no CUDA/cuDNN version matching**, and it falls back to
CPU when DirectML can't initialise. Measured ~1 s/page (GPU) vs ~3 s (CPU).

- Model input: 5-channel `[gray, hint(3)=0, mask(1)=0]`; grayscale in `[0,1]`;
  output de-normalised `out*0.5+0.5`. Short side scaled to `size` (÷32); tall
  webtoons are tiled (`smart_tiles`) and stitched.
- **`ort` uses `load-dynamic`** (see `Cargo.toml`) — pyke's `download-binaries`
  only ships a static CPU runtime, so real GPU needs an external ONNX Runtime.
- On first `/colorize`, `ensure_colorizer` fetches three assets (bundled next to
  the exe / in `models/`, else downloaded to the model-cache dir), points ort at
  the runtime via `ORT_DYLIB_PATH`, and adds its dir to `PATH` for `DirectML.dll`:

  | File | What | Size |
  |------|------|------|
  | `colorizer.onnx` | the model (5ch→3ch U-Net) | ~123 MB |
  | `onnxruntime.dll` | ONNX Runtime **1.22.x DirectML** build | ~16 MB |
  | `DirectML.dll` | DirectML runtime | ~18 MB |

  All three are hosted on Hugging Face (`HF_BASE` in `main.rs`). Re-export the
  model with `scripts/export_colorizer_onnx.py`.

> **Licensing:** manga-colorization-v2 has no explicit upstream license — the
> exported weights are hosted `other`/unknown with attribution. `onnxruntime.dll`
> is MIT; `DirectML.dll` is Microsoft-redistributable.

## MC-Trans integration (web + extension)

Two branch points, both "use helper if up, else WASM":

- **Web** (`src/lib/comic-service.ts`, `ocrImageBlob`): probe `GET /health`; if
  ok, `POST /ocr-page` and use the returned boxes; otherwise current WASM path.
- **Extension** (`extension/.../background.js`): `fetch` `127.0.0.1:7842`
  directly (host_permission bypasses CORS + the WASM-throttle slowdown).

## Build prerequisites (build machine only — end users need none of these)

- **MSVC Build Tools** (`cl.exe`, via `vcvars64.bat`) — nvcc's host compiler.
- **CUDA 13.x toolkit** (`nvcc`) for `--features cuda`. Match the major version of
  the runtime koharu downloads (CUDA **13**), so candle looks for `cublas64_13`.
- **cmake** on PATH (`winget install Kitware.CMake`).
- **LLVM / libclang** (`winget install LLVM.LLVM`) — `koharu-llm`'s build.rs runs
  bindgen to generate the llama.cpp FFI. Set `LIBCLANG_PATH=C:\Program Files\LLVM\bin`
  if bindgen can't find it.
- `.cargo/config.toml` sets `LLAMA_CPP_TAG` (the prebuilt llama.cpp release tag)
  and `CUDA_COMPUTE_CAP`.

All of the above are **build-time only**. The actual **llama.cpp + CUDA/Vulkan
runtime are downloaded prebuilt at runtime** (into the model-cache dir) to match
the user's hardware, exactly like koharu — never compiled, never bundled.
