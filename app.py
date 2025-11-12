import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from video_features.extractor import FeatureExtractionConfig, VideoFeatureExtractor, FarnebackParams, HistogramParams
from video_features.settings import AppSettings, ExtractorSettings

app_cfg = AppSettings.load()
ext_env = ExtractorSettings.load()

STATIC_DIR = app_cfg.static_dir
INDEX_HTML = STATIC_DIR / "index.html"

app_cfg.temp_volume_dir.mkdir(parents=True, exist_ok=True)

app = FastAPI(title=app_cfg.app_title)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _ensure_valid_upload(file: UploadFile) -> str:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in app_cfg.allowed_extensions:
        allowed = ", ".join(sorted(app_cfg.allowed_extensions))
        raise HTTPException(status_code=400, detail=f"Unsupported file extension '{ext}'. Allowed: {allowed}")
    if not (file.content_type or "").startswith(app_cfg.allowed_mime_prefix):
        raise HTTPException(status_code=400, detail=f"Unsupported content type '{file.content_type}'. Expected '{app_cfg.allowed_mime_prefix}*'.")
    return ext


def _dir_size_bytes(root: Path) -> int:
    total = 0
    for p in root.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


class _Quota:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.start = _dir_size_bytes(root)
        self.written = 0

    def check(self, n: int) -> None:
        if self.written + n > app_cfg.max_upload_bytes:
            raise HTTPException(status_code=413, detail=f"Upload exceeds {app_cfg.max_upload_bytes // (1024 * 1024)} MB limit.")
        projected = self.start + self.written + n
        if projected > app_cfg.volume_quota_bytes:
            raise HTTPException(status_code=429, detail="Temporary storage quota exceeded.")
        free_after = shutil.disk_usage(self.root).free - n
        if free_after < app_cfg.volume_min_free_bytes:
            raise HTTPException(status_code=507, detail="Insufficient free space on processing volume.")

    def add(self, n: int) -> None:
        self.written += n


async def _stream_upload_to_tempfile(upload: UploadFile, suffix: str) -> str:
    quota = _Quota(app_cfg.temp_volume_dir)
    tmp = tempfile.NamedTemporaryFile(dir=str(app_cfg.temp_volume_dir), prefix="vs_", suffix=suffix, delete=False)
    try:
        total = 0
        while True:
            chunk = await upload.read(app_cfg.upload_chunk_bytes)
            if not chunk:
                break
            quota.check(len(chunk))
            tmp.write(chunk)
            total += len(chunk)
            quota.add(len(chunk))
        if total == 0:
            raise HTTPException(status_code=400, detail="Empty upload.")
        return tmp.name
    finally:
        try:
            tmp.close()
        except Exception:
            pass
        try:
            await upload.close()
        except Exception:
            pass


def _build_feature_config() -> FeatureExtractionConfig:
    farneback = FarnebackParams(
        pyr_scale=ext_env.farneback_pyr_scale,
        levels=ext_env.farneback_levels,
        winsize=ext_env.farneback_winsize,
        iterations=ext_env.farneback_iterations,
        poly_n=ext_env.farneback_poly_n,
        poly_sigma=ext_env.farneback_poly_sigma,
        flags=ext_env.farneback_flags,
    )
    hist = HistogramParams(bins=ext_env.histogram_bins, ranges=ext_env.histogram_ranges)
    return FeatureExtractionConfig(
        frame_stride=ext_env.frame_stride,
        resize_width=ext_env.resize_width,
        shot_threshold=ext_env.shot_threshold,
        text_sample_stride=ext_env.text_sample_stride,
        text_min_chars=ext_env.text_min_chars,
        yolo_model=ext_env.yolo_model,
        yolo_frame_stride=ext_env.yolo_frame_stride,
        farneback=farneback,
        hist=hist,
    )


@app.get("/", response_class=HTMLResponse)
async def home() -> HTMLResponse:
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=404, detail="Missing static/index.html.")
    return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))


@app.get("/health")
def health_check() -> JSONResponse:
    return JSONResponse(content={"status": "healthy", "service": app_cfg.app_title}, status_code=200)


@app.post("/extract")
async def extract(file: UploadFile = File(...)) -> JSONResponse:
    ext = _ensure_valid_upload(file)
    temp_path = await _stream_upload_to_tempfile(file, suffix=ext)
    try:
        extractor = VideoFeatureExtractor(_build_feature_config())
        features = extractor.extract_features(temp_path)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Failed to read uploaded video file.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Video processing failed.") from exc
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
    return JSONResponse(content=features, status_code=200)
