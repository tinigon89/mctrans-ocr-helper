# mctrans-ocr-helper

Local OCR helper for MC-Trans comic translation. A thin HTTP server wrapping
[koharu-ml](https://github.com/mayocream/koharu)'s Candle models:

- **Detect** text regions — `ComicTextDetector`
- **Recognise** each region — **PaddleOCR-VL** (manga-capable VLM)
- **Inpaint** (optional) — **LaMa** removes original text → clean background

Runs on the user's own machine (GPU via CUDA/Metal, CPU fallback). The web app
and browser extension call it over loopback, so OCR runs natively instead of in
WASM. ~80 MB binary; model weights download from Hugging Face on first run.

> **License:** GPL-3.0 (links koharu-ml). Keep this in its **own repo** — do
> not vendor it into the proprietary MC-Trans web repo. The web app only talks
> to it over HTTP (separate process), so the web app stays proprietary.

## API

```
GET  /health                         -> { ok, name, gpu }
POST /ocr-page   (multipart)         -> { boxes: OcrBox[], cleanedImage? }
        file=<image bytes>           (required)
        inpaint=true                 (optional → also returns cleanedImage data URL)
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
Two OCR/ML backends both run on the GPU: candle (detection + inpaint) via the
CUDA toolkit, and llama.cpp (GGUF OCR) via its own bundled CUDA 13 DLLs.

Build prereqs: cmake, LLVM/libclang, **CUDA toolkit (nvcc)**, and **MSVC cl.exe**
(nvcc's host compiler). Build from a shell with the MSVC env loaded:

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "PATH=C:\Program Files\CMake\bin;C:\Program Files\LLVM\bin;%PATH%"
set "LIBCLANG_PATH=C:\Program Files\LLVM\bin"
cargo build --release --features cuda
```

Set `CUDA_COMPUTE_CAP` in `.cargo/config.toml` to your GPU (89 = RTX 4090 / Ada).

Run with `run.bat` (puts the CUDA toolkit's bin on PATH so candle finds
`cublas64_12.dll`).

> Two gotchas baked into the code/config:
> - **`[patch.crates-io]`** in `Cargo.toml` mirrors koharu's forked candle + ug
>   (cudarc `dynamic-loading`); without it the cuda build fails with "both
>   dynamic-loading and dynamic-linking active".
> - Candle (CUDA) models are loaded **before** `runtime.prepare()`, because that
>   call switches Windows to a restricted DLL search that hides the toolkit's
>   cublas from candle's cudarc.

### Apple Silicon
```bash
cargo run --release --features metal
```

Serves on `http://127.0.0.1:7842`. Override the model cache dir with
`MCTRANS_HELPER_ROOT`. ~64 MB main-thread stack is set in code (cuDNN DllMain is
stack-hungry).

Smoke test:

```bash
curl -s http://127.0.0.1:7842/health
curl -s -F file=@page.png -F inpaint=true http://127.0.0.1:7842/ocr-page | jq '.boxes | length'
```

## Distribution

`cargo build --release` → single binary (~80 MB). Ship via GitHub Releases per
OS. No Python, no PyTorch. Weights are fetched + cached on first launch.

## MC-Trans integration (web + extension)

Two branch points, both "use helper if up, else WASM":

- **Web** (`src/lib/comic-service.ts`, `ocrImageBlob`): probe `GET /health`; if
  ok, `POST /ocr-page` and use the returned boxes; otherwise current WASM path.
- **Extension** (`extension/.../background.js`): `fetch` `127.0.0.1:7842`
  directly (host_permission bypasses CORS + the WASM-throttle slowdown).

## Build prerequisites (build machine only — end users need none of these)

- **cmake** on PATH (`winget install Kitware.CMake`).
- **LLVM / libclang** (`winget install LLVM.LLVM`) — `koharu-llm`'s build.rs runs
  bindgen to generate the llama.cpp FFI. Set `LIBCLANG_PATH=C:\Program Files\LLVM\bin`
  if bindgen can't find it. This is **build-time only**; the shipped binary does
  not need it.
- `.cargo/config.toml` sets `LLAMA_CPP_TAG` (the prebuilt llama.cpp release tag).

The actual **llama.cpp + CUDA runtime are downloaded prebuilt at runtime** (into
the model-cache dir), exactly like koharu — never compiled here.

## Resolved during first build

- ✅ `ComputePolicy::PreferGpu` (GPU when available, else CPU).
- ✅ Models are `Send`/`Sync` — `Arc<Mutex<…>>` axum state compiles fine.

## Still to verify at runtime (with a real page)

- Whether `TextRegion` coords are pixels (assumed → divided by image w/h) or
  already fractional. Check box positions on a real OCR result.
- `bubble_mask` for LaMa — currently reuses the text mask; for cleaner inpaint
  feed a real `speech_bubble_segmentation` mask.
