//! MC-Trans local OCR helper — a thin HTTP wrapper around koharu-ml / koharu-llm.
//!
//! Endpoints (all loopback, 127.0.0.1):
//!   GET  /                 -> settings page (pick OCR + inpainter engines, etc.)
//!   GET  /health           -> { ok, name, device }
//!   GET  /config           -> current config (JSON)
//!   POST /config           -> save config (JSON body) to config.toml
//!   POST /ocr-page (multipart) -> { boxes: OcrBox[], cleanedImage? }
//!
//! Pipeline per page: chosen detector (PP-DocLayout V3 / Comic Text / Comic
//! Text & Bubble / Anime Text YOLO) -> chosen OCR per region -> optional
//! inpaint (chosen engine) -> text-removed PNG.
//!
//! GPL-3.0 (links koharu-ml/koharu-llm). Keep in its own repo, separate from
//! the proprietary MC-Trans web app (which only talks to it over HTTP).

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use axum::{
    Json, Router,
    extract::{Multipart, State},
    http::{HeaderValue, Request, StatusCode},
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use base64::Engine as _;
use image::DynamicImage;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, RwLock};
use tower_http::cors::CorsLayer;

use koharu_ml::aot_inpainting::AotInpainting;
use koharu_ml::anime_text::AnimeTextDetector;
use koharu_ml::comic_text_bubble_detector::ComicTextBubbleDetector;
use koharu_ml::comic_text_detector::ComicTextDetector;
use koharu_ml::lama::Lama;
use koharu_ml::manga_ocr::MangaOcr;
use koharu_ml::mit48px_ocr::Mit48pxOcr;
use koharu_ml::pp_doclayout_v3::PPDocLayoutV3;
use koharu_llm::paddleocr_vl::{PaddleOcrVl, PaddleOcrVlTask};
use koharu_llm::safe::llama_backend::LlamaBackend;
use koharu_runtime::{ComputePolicy, Runtime};

const BIND: &str = "127.0.0.1:7842";

// ---------------------------------------------------------------------------
// Config (persisted to <cache>/config.toml; editable from the settings page)
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct Config {
    /// "paddleocr-vl" | "manga-ocr" | "mit48px"   (restart to apply)
    ocr: String,
    /// "lama" | "aot" | "off"                      (restart to apply)
    inpainter: String,
    /// "gpu" | "cpu"                               (restart to apply)
    compute: String,
    /// text-block detector                         (restart to apply)
    /// "pp-doclayout" | "comic-text" | "comic-text-bubble" | "anime-text"
    detector: String,
    /// detection confidence threshold             (live)
    det_threshold: f32,
    /// "auto" | "ltr" | "rtl" reading order        (live)
    direction: String,
    /// inpaint by default when a request omits the flag (live)
    default_inpaint: bool,
    /// open a page in the default browser on startup (restart to apply)
    open_browser: bool,
    /// URL opened when open_browser is on (default = the helper settings page)
    open_url: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            ocr: "paddleocr-vl".into(),
            inpainter: "lama".into(),
            compute: "gpu".into(),
            detector: "pp-doclayout".into(),
            det_threshold: 0.3,
            direction: "auto".into(),
            default_inpaint: false,
            open_browser: true,
            open_url: "http://127.0.0.1:7842".into(),
        }
    }
}

fn config_path(root: &Path) -> PathBuf {
    root.join("config.toml")
}

fn load_config(root: &Path) -> Config {
    match std::fs::read_to_string(config_path(root)) {
        Ok(s) => toml::from_str(&s).unwrap_or_else(|e| {
            tracing::warn!("config.toml invalid ({e}); using defaults");
            Config::default()
        }),
        Err(_) => Config::default(),
    }
}

fn save_config(root: &Path, cfg: &Config) -> anyhow::Result<()> {
    std::fs::create_dir_all(root).ok();
    std::fs::write(config_path(root), toml::to_string_pretty(cfg)?)?;
    Ok(())
}

// ---- Data folder (model + runtime cache) location ----
// Config.toml lives INSIDE this folder, so its location can't live in the
// config. Resolution order: env var > pointer file next to the exe > default
// LOCALAPPDATA. The pointer file lets users move the cache off a full C: drive
// from the settings page without touching env vars.

fn data_dir_pointer() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("data-dir.txt")))
}

fn default_root() -> PathBuf {
    dirs::data_local_dir()
        .expect("no local data dir")
        .join("mctrans-ocr-helper")
}

fn env_root_override() -> Option<String> {
    std::env::var("MCTRANS_HELPER_ROOT").ok().filter(|v| !v.trim().is_empty())
}

fn resolve_root() -> PathBuf {
    if let Some(v) = env_root_override() {
        return PathBuf::from(v.trim());
    }
    if let Some(ptr) = data_dir_pointer() {
        if let Ok(s) = std::fs::read_to_string(&ptr) {
            let s = s.trim();
            if !s.is_empty() {
                return PathBuf::from(s);
            }
        }
    }
    default_root()
}

// ---- Self-update from GitHub Releases ----

const GH_OWNER: &str = "tinigon89";
const GH_REPO: &str = "mctrans-ocr-helper";
// Substring of the release asset name to download (mctrans-ocr-helper-windows-cuda.zip).
const UPDATE_TARGET: &str = "windows-cuda";

/// (current, latest) versions from GitHub. Blocking — run off the async runtime.
fn check_latest() -> anyhow::Result<(String, String)> {
    let current = env!("CARGO_PKG_VERSION").to_string();
    let releases = self_update::backends::github::ReleaseList::configure()
        .repo_owner(GH_OWNER)
        .repo_name(GH_REPO)
        .build()?
        .fetch()?;
    let latest = releases
        .first()
        .map(|r| r.version.clone())
        .unwrap_or_else(|| current.clone());
    Ok((current, latest))
}

/// Delete leftover `.mctrans-ocr-helper.<random>` temp files that self_replace
/// leaves next to the exe (the old binary, renamed during an update — it can't
/// be removed while it's the running process, so we sweep them on next launch).
fn cleanup_update_temp() {
    let dir = match std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf())) {
        Some(d) => d,
        None => return,
    };
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for e in entries.flatten() {
            let name = e.file_name();
            let name = name.to_string_lossy();
            // self_replace names its backups ".<exe stem>.<random>" (leading dot,
            // so the real "mctrans-ocr-helper.exe" is never matched).
            if name.starts_with(".mctrans-ocr-helper") {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }
}

/// Relaunch the (now-replaced) exe in a fresh console, then exit this process.
fn relaunch_and_exit() -> ! {
    if let Ok(exe) = std::env::current_exe() {
        let mut cmd = std::process::Command::new(&exe);
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NEW_CONSOLE: u32 = 0x0000_0010;
            cmd.creation_flags(CREATE_NEW_CONSOLE);
        }
        let _ = cmd.spawn();
    }
    std::process::exit(0);
}

/// Download the latest release + replace the running exe. Blocking. Returns the
/// version installed; the running process stays on the old code until restart.
fn run_self_update() -> anyhow::Result<String> {
    let status = self_update::backends::github::Update::configure()
        .repo_owner(GH_OWNER)
        .repo_name(GH_REPO)
        .bin_name("mctrans-ocr-helper")
        .target(UPDATE_TARGET)
        .bin_path_in_archive("mctrans-ocr-helper.exe")
        .current_version(env!("CARGO_PKG_VERSION"))
        .show_download_progress(false)
        .no_confirm(true)
        .build()?
        .update()?;
    Ok(status.version().to_string())
}

// ---------------------------------------------------------------------------
// Engine wrappers (one enum per swappable stage)
// ---------------------------------------------------------------------------

enum Ocr {
    PaddleVl(PaddleOcrVl),
    Manga(MangaOcr),
    Mit48px(Mit48pxOcr),
}

impl Ocr {
    fn recognize(&mut self, crop: &DynamicImage) -> anyhow::Result<String> {
        Ok(match self {
            Ocr::PaddleVl(m) => m.inference(crop, PaddleOcrVlTask::Ocr)?.text,
            Ocr::Manga(m) => m
                .inference(std::slice::from_ref(crop))?
                .into_iter()
                .next()
                .unwrap_or_default(),
            Ocr::Mit48px(m) => m
                .inference_regions(std::slice::from_ref(crop))?
                .into_iter()
                .next()
                .map(|p| p.text)
                .unwrap_or_default(),
        })
    }
}

enum Inpainter {
    Lama(Lama),
    Aot(AotInpainting),
    Off,
}

impl Inpainter {
    /// Returns None when disabled. `mask` = text to erase, `bubble_mask` = the
    /// region LaMa may fill from.
    fn run(
        &self,
        img: &DynamicImage,
        mask: &DynamicImage,
        bubble_mask: &DynamicImage,
    ) -> anyhow::Result<Option<DynamicImage>> {
        Ok(match self {
            Inpainter::Lama(m) => Some(m.inference(img, mask, bubble_mask)?),
            Inpainter::Aot(m) => Some(m.inference(img, mask, bubble_mask)?),
            Inpainter::Off => None,
        })
    }
}

/// Loaded models. Candle/llama inference is synchronous, so we serialise
/// requests behind one mutex (a local single-user helper does one page at a time).
struct Engines {
    layout: PPDocLayoutV3,        // PP-DocLayout V3 detection (always loaded)
    segmenter: ComicTextDetector, // segmentation MASK for inpaint (+ "comic-text" detector)
    bubble: Option<ComicTextBubbleDetector>, // loaded only when detector = comic-text-bubble
    anime: Option<AnimeTextDetector>,        // loaded only when detector = anime-text
    ocr: Ocr,
    inpainter: Inpainter,
}

#[derive(Clone)]
struct AppState {
    engines: Arc<Mutex<Engines>>,
    config: Arc<RwLock<Config>>,
    root: Arc<PathBuf>,
    device: &'static str,
}

// ---------------------------------------------------------------------------
// Response shapes
// ---------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct OcrBox {
    id: String,
    text: String,
    x: f32, // fractional [0,1] — matches MC-Trans OCRBox
    y: f32,
    width: f32,
    height: f32,
    confidence: f32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct OcrResponse {
    boxes: Vec<OcrBox>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cleaned_image: Option<String>,
}

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------

fn main() -> anyhow::Result<()> {
    // Windows threads default to a small (~1 MB) stack, which overflows while
    // loading the heavy cuDNN/CUDA DLLs. Run on a 64 MB stack (entry thread +
    // tokio worker/blocking threads, since model loading uses spawn_blocking).
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(|| {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .thread_stack_size(64 << 20)
                .build()?
                .block_on(async_main())
        })?
        .join()
        .expect("helper thread panicked")
}

async fn async_main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    // Sweep any leftover self-update temp files from a previous update.
    cleanup_update_temp();

    let root = resolve_root();
    std::fs::create_dir_all(&root).ok();
    tracing::info!("model cache: {}", root.display());

    let cfg = load_config(&root);
    let cpu = cfg.compute == "cpu";
    let device = if cpu { "cpu" } else { "gpu" };

    let runtime = Runtime::new(
        root.clone(),
        if cpu { ComputePolicy::CpuOnly } else { ComputePolicy::PreferGpu },
    )
    .context("failed to init koharu runtime")?;

    // Prepare the prebuilt runtime FIRST. On NVIDIA this downloads CUDA 13.1
    // (cudart/cublas64_13 …) + the llama.cpp CUDA build and preloads them; candle
    // (built against CUDA 13.1) then resolves the SAME cublas64_13 — no bundled
    // DLLs, no PATH. On non-NVIDIA it fetches the Vulkan build and candle falls
    // back to CPU. This mirrors koharu: the binary ships ~80 MB and the runtime
    // is fetched to match the user's hardware on first launch.
    tracing::info!("preparing runtime ({device})…");
    runtime.prepare().await.context("prepare runtime")?;

    tracing::info!("loading models (ocr={}, inpainter={})…", cfg.ocr, cfg.inpainter);
    let layout = PPDocLayoutV3::load(&runtime, cpu).await.context("load PP-DocLayout V3")?;
    let segmenter = ComicTextDetector::load(&runtime, cpu).await.context("load segmenter")?;

    // Optional detectors — loaded only when selected (each downloads its model).
    let bubble = if cfg.detector == "comic-text-bubble" {
        Some(
            ComicTextBubbleDetector::load(&runtime, cpu)
                .await
                .context("load Comic Text & Bubble detector")?,
        )
    } else {
        None
    };
    let anime = if cfg.detector == "anime-text" {
        Some(
            AnimeTextDetector::load(&runtime, cpu)
                .await
                .context("load Anime Text YOLO")?,
        )
    } else {
        None
    };
    tracing::info!("detector = {}", cfg.detector);

    let inpainter = match cfg.inpainter.as_str() {
        "off" => Inpainter::Off,
        "aot" => Inpainter::Aot(AotInpainting::load(&runtime, cpu).await.context("load AOT")?),
        _ => Inpainter::Lama(Lama::load(&runtime, cpu).await.context("load LaMa")?),
    };

    let ocr = match cfg.ocr.as_str() {
        "manga-ocr" => Ocr::Manga(MangaOcr::load(&runtime, cpu).await.context("load Manga OCR")?),
        "mit48px" => Ocr::Mit48px(Mit48pxOcr::load(&runtime, cpu).await.context("load MIT 48px")?),
        _ => {
            // PaddleOCR-VL 1.6 GGUF via llama.cpp (runtime already prepared above).
            koharu_llm::sys::initialize(&runtime).context("init llama.cpp bindings")?;
            let backend = Arc::new(
                LlamaBackend::init().map_err(|e| anyhow::anyhow!("llama backend init: {e:?}"))?,
            );
            Ocr::PaddleVl(
                PaddleOcrVl::load(&runtime, cpu, backend)
                    .await
                    .context("load PaddleOCR-VL 1.6 GGUF")?,
            )
        }
    };
    tracing::info!("models ready");

    // Capture the browser-open settings before `cfg` is moved into the state.
    let open_browser = cfg.open_browser;
    let open_url = cfg.open_url.clone();

    let state = AppState {
        engines: Arc::new(Mutex::new(Engines { layout, segmenter, bubble, anime, ocr, inpainter })),
        config: Arc::new(RwLock::new(cfg)),
        root: Arc::new(root),
        device,
    };

    let app = Router::new()
        .route("/", get(settings_page))
        .route("/health", get(health))
        .route("/config", get(get_config).post(post_config))
        // Staged endpoints (single-page editor) + one-shot (batch).
        .route("/detect", post(detect_handler))
        .route("/ocr", post(ocr_handler))
        .route("/inpaint", post(inpaint_handler))
        .route("/ocr-page", post(ocr_page))
        .route("/data-dir", get(get_data_dir).post(set_data_dir))
        .route("/update-check", get(update_check))
        .route("/update", post(update_apply))
        .layer(CorsLayer::very_permissive())
        .layer(middleware::from_fn(add_pna_header))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(BIND).await?;
    tracing::info!("MC-Trans OCR helper on http://{BIND}  (settings: open it in a browser)");
    // Open the web app in the default browser now that the server is listening.
    if open_browser && !open_url.trim().is_empty() {
        tracing::info!("opening {open_url} in the default browser");
        if let Err(e) = open::that(open_url.trim()) {
            tracing::warn!("could not open browser: {e}");
        }
    }
    // Background update check (non-blocking) — just logs; the user applies it
    // from the settings page. Network failures are ignored.
    tokio::spawn(async {
        if let Ok(Ok((current, latest))) = tokio::task::spawn_blocking(check_latest).await {
            if self_update::version::bump_is_greater(&current, &latest).unwrap_or(false) {
                tracing::info!("update available: v{current} -> v{latest} (open settings to update)");
            }
        }
    });
    axum::serve(listener, app).await?;
    Ok(())
}

/// PNA header so secure (https) pages can fetch this loopback server.
async fn add_pna_header(req: Request<axum::body::Body>, next: Next) -> Response {
    let mut res = next.run(req).await;
    res.headers_mut().insert(
        "Access-Control-Allow-Private-Network",
        HeaderValue::from_static("true"),
    );
    res
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn health(State(s): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "ok": true,
        "name": "mctrans-ocr-helper",
        "version": env!("CARGO_PKG_VERSION"),
        "device": s.device,
    }))
}

async fn get_config(State(s): State<AppState>) -> Json<Config> {
    Json(s.config.read().await.clone())
}

async fn post_config(
    State(s): State<AppState>,
    Json(new_cfg): Json<Config>,
) -> Result<Json<serde_json::Value>, AppError> {
    save_config(&s.root, &new_cfg).map_err(AppError::internal)?;
    *s.config.write().await = new_cfg;
    Ok(Json(serde_json::json!({ "ok": true })))
}

async fn get_data_dir(State(s): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "root": s.root.display().to_string(),
        "default": default_root().display().to_string(),
        // When the env var forces the location, the settings field is read-only.
        "envOverride": env_root_override(),
    }))
}

#[derive(serde::Deserialize)]
struct DataDirBody {
    /// New data folder; empty string reverts to the default location.
    path: String,
}

async fn set_data_dir(Json(body): Json<DataDirBody>) -> Result<Json<serde_json::Value>, AppError> {
    if env_root_override().is_some() {
        return Err(AppError::bad(anyhow::anyhow!(
            "data folder is pinned by the MCTRANS_HELPER_ROOT environment variable"
        )));
    }
    let ptr = data_dir_pointer()
        .ok_or_else(|| AppError::internal(anyhow::anyhow!("cannot locate the exe directory")))?;
    let p = body.path.trim();
    if p.is_empty() {
        // Revert to the default location.
        let _ = std::fs::remove_file(&ptr);
    } else {
        // Create it now so an unwritable / invalid path fails here, not on the
        // next launch. Then persist the pointer next to the exe.
        std::fs::create_dir_all(p).map_err(|e| AppError::bad(anyhow::anyhow!("cannot use that folder: {e}")))?;
        std::fs::write(&ptr, p).map_err(AppError::internal)?;
    }
    Ok(Json(serde_json::json!({ "ok": true })))
}

async fn update_check() -> Result<Json<serde_json::Value>, AppError> {
    let (current, latest) = tokio::task::spawn_blocking(check_latest)
        .await
        .map_err(AppError::internal)?
        .map_err(AppError::internal)?;
    let available = self_update::version::bump_is_greater(&current, &latest).unwrap_or(false);
    Ok(Json(serde_json::json!({ "current": current, "latest": latest, "available": available })))
}

async fn update_apply() -> Result<Json<serde_json::Value>, AppError> {
    let version = tokio::task::spawn_blocking(run_self_update)
        .await
        .map_err(AppError::internal)?
        .map_err(AppError::internal)?;
    // Auto-restart: let the HTTP response flush, then relaunch the new exe in a
    // fresh console and exit this (old) process.
    tokio::spawn(async {
        tokio::time::sleep(std::time::Duration::from_millis(800)).await;
        tracing::info!("update installed — restarting");
        relaunch_and_exit();
    });
    Ok(Json(serde_json::json!({ "ok": true, "version": version, "restarting": true })))
}

async fn settings_page() -> Html<&'static str> {
    Html(SETTINGS_HTML)
}

/// POST /ocr-page (multipart): file=<image>, inpaint=true|false (optional override).
async fn ocr_page(
    State(s): State<AppState>,
    mut form: Multipart,
) -> Result<Json<OcrResponse>, AppError> {
    let mut img_bytes: Option<Vec<u8>> = None;
    let mut inpaint_override: Option<bool> = None;
    let mut direction_override: Option<String> = None;

    while let Some(field) = form.next_field().await.map_err(AppError::bad)? {
        match field.name() {
            Some("file") => img_bytes = Some(field.bytes().await.map_err(AppError::bad)?.to_vec()),
            Some("inpaint") => {
                let v = field.text().await.map_err(AppError::bad)?;
                inpaint_override = Some(v == "true" || v == "1");
            }
            Some("direction") => {
                direction_override = Some(field.text().await.map_err(AppError::bad)?);
            }
            _ => {}
        }
    }

    let bytes = img_bytes.ok_or_else(|| AppError::bad("missing 'file' field"))?;
    let img = image::load_from_memory(&bytes).map_err(AppError::bad)?;
    let (iw, ih) = (img.width() as f32, img.height() as f32);

    let cfg = s.config.read().await.clone();
    let want_inpaint = inpaint_override.unwrap_or(cfg.default_inpaint);
    // Per-request direction (from the web's reading-order setting) overrides config.
    let direction = direction_override.as_deref().unwrap_or(&cfg.direction);
    let mut engines = s.engines.lock().await;

    // 1. Detect text blocks, then OCR each crop.
    let regions = detect_regions(&engines, &img, cfg.det_threshold, direction, &cfg.detector)
        .map_err(AppError::internal)?;
    let mut boxes = Vec::new();
    for (bbox, score) in regions {
        let text = ocr_crop(&mut engines, &img, bbox).map_err(AppError::internal)?;
        if text.is_empty() {
            continue;
        }
        let [x1, y1, x2, y2] = bbox;
        boxes.push(OcrBox {
            id: nanoid::nanoid!(10),
            text,
            x: x1 / iw,
            y: y1 / ih,
            width: (x2 - x1) / iw,
            height: (y2 - y1) / ih,
            confidence: score,
        });
    }

    // 3. Optional inpaint.
    let cleaned_image = if want_inpaint {
        let (mask, region) =
            inpaint_masks(&engines, &img, cfg.det_threshold, &cfg.detector).map_err(AppError::internal)?;
        match engines.inpainter.run(&img, &mask, &region).map_err(AppError::internal)? {
            Some(clean) => Some(to_data_url_png(&clean).map_err(AppError::internal)?),
            None => None,
        }
    } else {
        None
    };

    Ok(Json(OcrResponse { boxes, cleaned_image }))
}

/// Row-bucketed reading-order key. `rtl` flips the horizontal direction.
fn order_key_bbox(bbox: [f32; 4], rtl: bool) -> f32 {
    let [x1, y1, _, _] = bbox;
    let row = (y1 / 40.0).floor() * 100_000.0; // ~40px rows, dominates the key
    row + if rtl { -x1 } else { x1 }
}

fn to_data_url_png(img: &DynamicImage) -> anyhow::Result<String> {
    let mut buf = Vec::new();
    img.write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png)?;
    Ok(format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(&buf)))
}

// ---------------------------------------------------------------------------
// Shared pipeline pieces (used by both one-shot /ocr-page and staged endpoints)
// ---------------------------------------------------------------------------

/// Detect text-block pixel bboxes ([x1,y1,x2,y2]) + score, filtered & ordered.
/// `detector` picks the model: pp-doclayout (default) / comic-text /
/// comic-text-bubble / anime-text.
fn detect_regions(
    engines: &Engines,
    img: &DynamicImage,
    threshold: f32,
    direction: &str,
    detector: &str,
) -> anyhow::Result<Vec<([f32; 4], f32)>> {
    let (iw, ih) = (img.width() as f32, img.height() as f32);
    // PP-DocLayout predicts a reading order we honour for "auto"; the other
    // detectors don't, so we fall back to a bbox-based sort there.
    let mut pp_auto = false;
    let mut out: Vec<([f32; 4], f32)> = match detector {
        "comic-text" => engines
            .segmenter
            .inference(img)?
            .text_blocks
            .into_iter()
            .map(|r| ([r.x, r.y, r.x + r.width, r.y + r.height], r.confidence))
            .collect(),
        "comic-text-bubble" => engines
            .bubble
            .as_ref()
            .context("Comic Text & Bubble detector not loaded — restart the helper")?
            .inference_with_threshold(img, threshold)?
            .text_blocks
            .into_iter()
            .map(|r| ([r.x, r.y, r.x + r.width, r.y + r.height], r.confidence))
            .collect(),
        "anime-text" => engines
            .anime
            .as_ref()
            .context("Anime Text YOLO not loaded — restart the helper")?
            .inference(img)?
            .regions
            .into_iter()
            .map(|r| (r.bbox, r.score))
            .collect(),
        _ => {
            let mut regions = engines.layout.inference_one(img, threshold)?.regions;
            if direction == "auto" {
                regions.sort_by_key(|r| r.order);
                pp_auto = true;
            }
            regions.into_iter().map(|r| (r.bbox, r.score)).collect()
        }
    };
    // Drop degenerate / panel-sized regions (likely figures, not text).
    out.retain(|(bbox, _)| {
        let [x1, y1, x2, y2] = *bbox;
        let (rw, rh) = (x2 - x1, y2 - y1);
        rw >= 3.0 && rh >= 3.0 && (rw * rh) <= 0.5 * iw * ih
    });
    if !pp_auto {
        let rtl = direction == "rtl";
        out.sort_by(|a, b| order_key_bbox(a.0, rtl).total_cmp(&order_key_bbox(b.0, rtl)));
    }
    Ok(out)
}

/// Crop a pixel bbox and run the configured OCR engine; trimmed text.
fn ocr_crop(engines: &mut Engines, img: &DynamicImage, bbox: [f32; 4]) -> anyhow::Result<String> {
    let [x1, y1, x2, y2] = bbox;
    let crop = img.crop_imm(
        x1.max(0.0) as u32,
        y1.max(0.0) as u32,
        (x2 - x1) as u32,
        (y2 - y1) as u32,
    );
    Ok(engines.ocr.recognize(&crop)?.trim().to_string())
}

/// Build the two masks LaMa wants (koharu-style), both at image resolution:
///   - text mask: segmentation strokes CONSTRAINED to detected text-block boxes
///     (so hair / dark sketchy art the segmenter false-positives on is never
///     erased — only text *inside* a detected text region is removed), dilated.
///   - region mask: the text-box rectangles, passed as LaMa's `bubble_mask` so
///     it fills the text from the surrounding region (koharu uses the speech-
///     bubble segmentation here; the boxes are a good approximation).
fn inpaint_masks(
    engines: &Engines,
    img: &DynamicImage,
    threshold: f32,
    detector: &str,
) -> anyhow::Result<(DynamicImage, DynamicImage)> {
    // Refined text mask (UNet + DBNet) — cleaner strokes than raw `inference_segmentation`.
    let seg = engines.segmenter.inference(img)?.mask;
    let (w, h) = (seg.width(), seg.height());
    let (iw, ih) = (img.width() as f32, img.height() as f32);
    let (sx, sy) = (w as f32 / iw, h as f32 / ih); // mask may differ from page res

    // Constrain to PP-DocLayout text boxes — better coverage than the segmenter's
    // own boxes (catches signs like 保健室) while still excluding hair / art, so
    // those are never erased. Box rectangles double as LaMa's bubble_mask.
    let regions = detect_regions(engines, img, threshold, "auto", detector)?;
    let mut region_mask = image::GrayImage::new(w, h);
    for ([x1, y1, x2, y2], _) in &regions {
        let pad = 6.0;
        let x0 = (((x1 - pad) * sx).max(0.0)) as u32;
        let y0 = (((y1 - pad) * sy).max(0.0)) as u32;
        let x3 = (((x2 + pad) * sx).min(w as f32)) as u32;
        let y3 = (((y2 + pad) * sy).min(h as f32)) as u32;
        for yy in y0..y3 {
            for xx in x0..x3 {
                region_mask.put_pixel(xx, yy, image::Luma([255]));
            }
        }
    }

    // Keep only text strokes that fall inside a detected text box.
    let mut text = image::GrayImage::new(w, h);
    for (x, y, p) in seg.enumerate_pixels() {
        if p[0] > 127 && region_mask.get_pixel(x, y)[0] > 0 {
            text.put_pixel(x, y, image::Luma([255]));
        }
    }
    let text = imageproc::morphology::dilate(&text, imageproc::distance_transform::Norm::LInf, 2);
    Ok((DynamicImage::ImageLuma8(text), DynamicImage::ImageLuma8(region_mask)))
}

/// Pull the `file` field out of a multipart form as a decoded image.
async fn read_image(form: &mut Multipart) -> Result<DynamicImage, AppError> {
    let mut bytes = None;
    while let Some(field) = form.next_field().await.map_err(AppError::bad)? {
        if field.name() == Some("file") {
            bytes = Some(field.bytes().await.map_err(AppError::bad)?.to_vec());
        }
    }
    let bytes = bytes.ok_or_else(|| AppError::bad("missing 'file' field"))?;
    image::load_from_memory(&bytes).map_err(AppError::bad)
}

// ---------------------------------------------------------------------------
// Staged endpoints — for the single-page editor (koharu-style layers)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeomBox {
    id: String,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    confidence: f32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DetectResponse {
    boxes: Vec<GeomBox>,
    /// Segmentation mask PNG (white = text pixels), for the mask layer + inpaint.
    mask: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct BoxIn {
    #[serde(default)]
    id: Option<String>,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct InpaintResponse {
    cleaned_image: String,
}

/// POST /detect (file) — boxes + segmentation mask (koharu's "Detect" = both).
async fn detect_handler(
    State(s): State<AppState>,
    mut form: Multipart,
) -> Result<Json<DetectResponse>, AppError> {
    let mut img_bytes: Option<Vec<u8>> = None;
    let mut direction_override: Option<String> = None;
    while let Some(field) = form.next_field().await.map_err(AppError::bad)? {
        match field.name() {
            Some("file") => img_bytes = Some(field.bytes().await.map_err(AppError::bad)?.to_vec()),
            Some("direction") => {
                direction_override = Some(field.text().await.map_err(AppError::bad)?)
            }
            _ => {}
        }
    }
    let bytes = img_bytes.ok_or_else(|| AppError::bad("missing 'file' field"))?;
    let img = image::load_from_memory(&bytes).map_err(AppError::bad)?;
    let (iw, ih) = (img.width() as f32, img.height() as f32);
    let cfg = s.config.read().await.clone();
    let direction = direction_override.as_deref().unwrap_or(&cfg.direction);
    let engines = s.engines.lock().await;

    let regions = detect_regions(&engines, &img, cfg.det_threshold, direction, &cfg.detector)
        .map_err(AppError::internal)?;
    let boxes = regions
        .into_iter()
        .map(|([x1, y1, x2, y2], score)| GeomBox {
            id: nanoid::nanoid!(10),
            x: x1 / iw,
            y: y1 / ih,
            width: (x2 - x1) / iw,
            height: (y2 - y1) / ih,
            confidence: score,
        })
        .collect();

    // Same refined mask the inpaint uses (UNet + DBNet), so the LAYERS view
    // matches what gets erased.
    let mask = DynamicImage::ImageLuma8(
        engines.segmenter.inference(&img).map_err(AppError::internal)?.mask,
    );
    Ok(Json(DetectResponse {
        boxes,
        mask: to_data_url_png(&mask).map_err(AppError::internal)?,
    }))
}

/// POST /ocr (file + boxes JSON) — OCR exactly the given (possibly edited) boxes.
async fn ocr_handler(
    State(s): State<AppState>,
    mut form: Multipart,
) -> Result<Json<OcrResponse>, AppError> {
    let mut img_bytes: Option<Vec<u8>> = None;
    let mut boxes_json: Option<String> = None;
    while let Some(field) = form.next_field().await.map_err(AppError::bad)? {
        match field.name() {
            Some("file") => img_bytes = Some(field.bytes().await.map_err(AppError::bad)?.to_vec()),
            Some("boxes") => boxes_json = Some(field.text().await.map_err(AppError::bad)?),
            _ => {}
        }
    }
    let bytes = img_bytes.ok_or_else(|| AppError::bad("missing 'file' field"))?;
    let img = image::load_from_memory(&bytes).map_err(AppError::bad)?;
    let (iw, ih) = (img.width() as f32, img.height() as f32);
    let boxes_in: Vec<BoxIn> =
        serde_json::from_str(&boxes_json.ok_or_else(|| AppError::bad("missing 'boxes' field"))?)
            .map_err(AppError::bad)?;

    let mut engines = s.engines.lock().await;
    let mut out = Vec::new();
    for b in boxes_in {
        let bbox = [b.x * iw, b.y * ih, (b.x + b.width) * iw, (b.y + b.height) * ih];
        let text = ocr_crop(&mut engines, &img, bbox).map_err(AppError::internal)?;
        out.push(OcrBox {
            id: b.id.unwrap_or_else(|| nanoid::nanoid!(10)),
            text,
            x: b.x,
            y: b.y,
            width: b.width,
            height: b.height,
            confidence: 1.0,
        });
    }
    Ok(Json(OcrResponse { boxes: out, cleaned_image: None }))
}

/// POST /inpaint (file) — segmentation mask → inpaint → text-removed PNG.
async fn inpaint_handler(
    State(s): State<AppState>,
    mut form: Multipart,
) -> Result<Json<InpaintResponse>, AppError> {
    let img = read_image(&mut form).await?;
    let (threshold, detector) = {
        let c = s.config.read().await;
        (c.det_threshold, c.detector.clone())
    };
    let engines = s.engines.lock().await;
    let (mask, region) = inpaint_masks(&engines, &img, threshold, &detector).map_err(AppError::internal)?;
    match engines.inpainter.run(&img, &mask, &region).map_err(AppError::internal)? {
        Some(clean) => Ok(Json(InpaintResponse {
            cleaned_image: to_data_url_png(&clean).map_err(AppError::internal)?,
        })),
        None => Err(AppError::bad("inpainter is set to Off")),
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

struct AppError(StatusCode, String);
impl AppError {
    fn bad(e: impl ToString) -> Self {
        AppError(StatusCode::BAD_REQUEST, e.to_string())
    }
    fn internal(e: impl ToString) -> Self {
        AppError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    }
}
impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (self.0, Json(serde_json::json!({ "error": self.1 }))).into_response()
    }
}

const SETTINGS_HTML: &str = include_str!("settings.html");
