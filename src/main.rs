//! MC-Trans local OCR helper — a thin HTTP wrapper around koharu-ml / koharu-llm.
//!
//! Endpoints (all loopback, 127.0.0.1):
//!   GET  /                 -> settings page (pick OCR + inpainter engines, etc.)
//!   GET  /health           -> { ok, name, device }
//!   GET  /config           -> current config (JSON)
//!   POST /config           -> save config (JSON body) to config.toml
//!   POST /ocr-page (multipart) -> { boxes: OcrBox[], cleanedImage? }
//!
//! Pipeline per page: PP-DocLayout V3 (detect) -> chosen OCR per region ->
//! optional inpaint (chosen engine) -> text-removed PNG.
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
    /// detection confidence threshold             (live)
    det_threshold: f32,
    /// "auto" | "ltr" | "rtl" reading order        (live)
    direction: String,
    /// inpaint by default when a request omits the flag (live)
    default_inpaint: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            ocr: "paddleocr-vl".into(),
            inpainter: "lama".into(),
            compute: "gpu".into(),
            det_threshold: 0.3,
            direction: "auto".into(),
            default_inpaint: false,
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
    /// Returns None when disabled. mask is reused as the bubble mask.
    fn run(&self, img: &DynamicImage, mask: &DynamicImage) -> anyhow::Result<Option<DynamicImage>> {
        Ok(match self {
            Inpainter::Lama(m) => Some(m.inference(img, mask, mask)?),
            Inpainter::Aot(m) => Some(m.inference(img, mask, mask)?),
            Inpainter::Off => None,
        })
    }
}

/// Loaded models. Candle/llama inference is synchronous, so we serialise
/// requests behind one mutex (a local single-user helper does one page at a time).
struct Engines {
    layout: PPDocLayoutV3,     // text-block DETECTION (fixed)
    segmenter: ComicTextDetector, // segmentation MASK for inpaint
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

    let root = std::env::var("MCTRANS_HELPER_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::data_local_dir()
                .expect("no local data dir")
                .join("mctrans-ocr-helper")
        });
    tracing::info!("model cache: {}", root.display());

    let cfg = load_config(&root);
    let cpu = cfg.compute == "cpu";
    let device = if cpu { "cpu" } else { "gpu" };

    let runtime = Runtime::new(
        root.clone(),
        if cpu { ComputePolicy::CpuOnly } else { ComputePolicy::PreferGpu },
    )
    .context("failed to init koharu runtime")?;

    tracing::info!("loading models (ocr={}, inpainter={}, {})…", cfg.ocr, cfg.inpainter, device);

    // --- Candle models load FIRST, before any llama `prepare()` switches Windows
    //     to a restricted DLL search that would hide the CUDA toolkit's cublas. ---
    let layout = PPDocLayoutV3::load(&runtime, cpu).await.context("load PP-DocLayout V3")?;
    let segmenter = ComicTextDetector::load(&runtime, cpu).await.context("load segmenter")?;

    let inpainter = match cfg.inpainter.as_str() {
        "off" => Inpainter::Off,
        "aot" => Inpainter::Aot(AotInpainting::load(&runtime, cpu).await.context("load AOT")?),
        _ => Inpainter::Lama(Lama::load(&runtime, cpu).await.context("load LaMa")?),
    };

    let ocr = match cfg.ocr.as_str() {
        "manga-ocr" => Ocr::Manga(MangaOcr::load(&runtime, cpu).await.context("load Manga OCR")?),
        "mit48px" => Ocr::Mit48px(Mit48pxOcr::load(&runtime, cpu).await.context("load MIT 48px")?),
        _ => {
            // PaddleOCR-VL 1.6 GGUF via llama.cpp — set up the llama runtime now
            // (after all candle models have grabbed their CUDA libs).
            tracing::info!("preparing llama.cpp runtime…");
            runtime.prepare().await.context("prepare llama runtime")?;
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

    let state = AppState {
        engines: Arc::new(Mutex::new(Engines { layout, segmenter, ocr, inpainter })),
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
        .layer(CorsLayer::very_permissive())
        .layer(middleware::from_fn(add_pna_header))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(BIND).await?;
    tracing::info!("MC-Trans OCR helper on http://{BIND}  (settings: open it in a browser)");
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
    Json(serde_json::json!({ "ok": true, "name": "mctrans-ocr-helper", "device": s.device }))
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

    while let Some(field) = form.next_field().await.map_err(AppError::bad)? {
        match field.name() {
            Some("file") => img_bytes = Some(field.bytes().await.map_err(AppError::bad)?.to_vec()),
            Some("inpaint") => {
                let v = field.text().await.map_err(AppError::bad)?;
                inpaint_override = Some(v == "true" || v == "1");
            }
            _ => {}
        }
    }

    let bytes = img_bytes.ok_or_else(|| AppError::bad("missing 'file' field"))?;
    let img = image::load_from_memory(&bytes).map_err(AppError::bad)?;
    let (iw, ih) = (img.width() as f32, img.height() as f32);

    let cfg = s.config.read().await.clone();
    let want_inpaint = inpaint_override.unwrap_or(cfg.default_inpaint);
    let mut engines = s.engines.lock().await;

    // 1. Detect text blocks, then OCR each crop.
    let regions = detect_regions(&engines, &img, cfg.det_threshold, &cfg.direction)
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
        let mask = DynamicImage::ImageLuma8(
            engines.segmenter.inference_segmentation(&img).map_err(AppError::internal)?,
        );
        match engines.inpainter.run(&img, &mask).map_err(AppError::internal)? {
            Some(clean) => Some(to_data_url_png(&clean).map_err(AppError::internal)?),
            None => None,
        }
    } else {
        None
    };

    Ok(Json(OcrResponse { boxes, cleaned_image }))
}

/// Row-bucketed reading-order key. `rtl` flips the horizontal direction.
fn order_key(r: &koharu_ml::pp_doclayout_v3::LayoutRegion, rtl: bool) -> f32 {
    let [x1, y1, _, _] = r.bbox;
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
fn detect_regions(
    engines: &Engines,
    img: &DynamicImage,
    threshold: f32,
    direction: &str,
) -> anyhow::Result<Vec<([f32; 4], f32)>> {
    let (iw, ih) = (img.width() as f32, img.height() as f32);
    let mut regions = engines.layout.inference_one(img, threshold)?.regions;
    match direction {
        "ltr" => regions.sort_by(|a, b| order_key(a, false).total_cmp(&order_key(b, false))),
        "rtl" => regions.sort_by(|a, b| order_key(a, true).total_cmp(&order_key(b, true))),
        _ => regions.sort_by_key(|r| r.order), // "auto" — PP-DocLayout's order
    }
    let mut out = Vec::new();
    for r in &regions {
        let [x1, y1, x2, y2] = r.bbox;
        let (rw, rh) = (x2 - x1, y2 - y1);
        // Skip degenerate / panel-sized regions (likely figures, not text).
        if rw < 3.0 || rh < 3.0 || (rw * rh) > 0.5 * iw * ih {
            continue;
        }
        out.push(([x1, y1, x2, y2], r.score));
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
    let img = read_image(&mut form).await?;
    let (iw, ih) = (img.width() as f32, img.height() as f32);
    let cfg = s.config.read().await.clone();
    let engines = s.engines.lock().await;

    let regions = detect_regions(&engines, &img, cfg.det_threshold, &cfg.direction)
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

    let mask = DynamicImage::ImageLuma8(
        engines.segmenter.inference_segmentation(&img).map_err(AppError::internal)?,
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
    let engines = s.engines.lock().await;
    let mask = DynamicImage::ImageLuma8(
        engines.segmenter.inference_segmentation(&img).map_err(AppError::internal)?,
    );
    match engines.inpainter.run(&img, &mask).map_err(AppError::internal)? {
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
