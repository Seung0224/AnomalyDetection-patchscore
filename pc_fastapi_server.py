# pc_fastapi_server.py
import os
import io
import sys
import time
import base64
import logging
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np
import torch
from fastapi import FastAPI, Request, UploadFile, File, Form  # ← 멀티파트용 import

# ---- (옵션) FAISS 임포트: 없으면 None 처리 ----
try:
    import faiss  # faiss-cpu / faiss-gpu
except Exception:
    faiss = None

# === 레포 모듈 ===
import inference
from overlay_utils import _predict, _make_overlay
from model_loader import (load_model, load_model_from_files,
                          validate_model_dir, validate_model_files)

# --------------------------
# 환경설정
# --------------------------
DEFAULT_MODEL_DIR = r"D:\ADI\patchcore\models\Cable_OK_ROI_SPEED"
HOST   = os.getenv("PC_HOST", "127.0.0.1")
PORT   = int(os.getenv("PC_PORT", "8009"))
DEVICE = os.getenv("PC_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

ENV_FAISS = os.getenv("PC_FAISS", os.path.join(DEFAULT_MODEL_DIR, "nnscorer_search_index.faiss"))
ENV_PKL   = os.getenv("PC_PKL",   os.path.join(DEFAULT_MODEL_DIR, "patchcore_params.pkl"))
ENV_DIR   = os.getenv("PC_MODEL_DIR", DEFAULT_MODEL_DIR)

# --------------------------
# 로깅 (콘솔)
# --------------------------
logger = logging.getLogger("pc_server")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
logger.addHandler(_handler)

# --------------------------
# FastAPI 앱
# --------------------------
app = FastAPI(title="PatchCore FastAPI (FAISS+PKL Bridge)")

MODEL = None
MODEL_DEVICE = DEVICE

# --------------------------
# 요청/응답 모델
# --------------------------
class InferReq(BaseModel):
    image_path: Optional[str] = None
    image_b64: Optional[str] = None
    # overlay params (python 기본값)
    clip_q: float = 0.98
    alpha_min: float = 0.02
    alpha_max: float = 0.5
    gamma: float = 1.8
    blur: float = 0.2
    # 성능 옵션
    return_images: bool = True                 # False면 후처리/인코딩 모두 생략
    overlay_format: str = "jpg"                # "png" | "jpg"
    overlay_quality: int = 90                  # jpg일 때 품질(1~100)
    overlay_max_side: int = 1280               # 응답 이미지 최대 변 길이(0=원본)

class InferRes(BaseModel):
    ok: bool
    score: float
    is_ng: bool
    heatmap_png_b64: str
    overlay_png_b64: str
    proc_ms: float       # 전체 처리 시간 (전처리/추론/후처리/인코딩 모두)
    infer_ms: float      # 순수 추론 시간 (_predict 만)
    post_ms: float       # 후처리/인코딩(그림 생성+압축) 시간

# --------------------------
# 유틸
# --------------------------
def _pil_from_req(req: InferReq) -> Image.Image:
    if req.image_path:
        return Image.open(req.image_path).convert("RGB")
    if req.image_b64:
        raw = base64.b64decode(req.image_b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    raise ValueError("image_path 또는 image_b64 중 하나는 필수입니다.")

def _tune_faiss_threads():
    """FAISS OMP 스레드 수를 환경변수로 설정 (기본: CPU 코어 수)"""
    threads_env = os.getenv("PC_FAISS_THREADS")
    try:
        default_threads = os.cpu_count() or 8
    except Exception:
        default_threads = 8
    threads = int(threads_env) if threads_env else default_threads

    if faiss is not None:
        try:
            faiss.omp_set_num_threads(threads)
            max_thr = faiss.omp_get_max_threads() if hasattr(faiss, "omp_get_max_threads") else threads
            logger.info(f"[FAISS] omp threads = {max_thr}")
        except Exception as e:
            logger.warning(f"[FAISS] set threads failed: {e}")
    else:
        logger.warning("[FAISS] module not available; skip thread tuning")

def _load_model_once():
    global MODEL, MODEL_DEVICE
    if MODEL is not None:
        return MODEL

    # 파일 경로 지정된 경우
    if os.path.isfile(ENV_FAISS) and os.path.isfile(ENV_PKL):
        ok, msg = validate_model_files(ENV_FAISS, ENV_PKL)
        if not ok:
            raise FileNotFoundError(msg)
        logger.info(f"[MODEL] load from files: FAISS={ENV_FAISS} | PKL={ENV_PKL} | device={DEVICE}")
        MODEL, MODEL_DEVICE = load_model_from_files(ENV_FAISS, ENV_PKL, device=DEVICE, reset_cache=True)
        logger.info("[MODEL] loaded (files).")
        return MODEL

    # 디렉터리 기준
    ok, msg = validate_model_dir(ENV_DIR)
    if not ok:
        raise FileNotFoundError(msg)
    logger.info(f"[MODEL] load from dir: {ENV_DIR} | device={DEVICE}")
    MODEL, MODEL_DEVICE = load_model(ENV_DIR, device=DEVICE, reset_cache=True)
    logger.info("[MODEL] loaded (dir).")
    return MODEL

def _resize_for_response(pil: Image.Image, max_side: int) -> Image.Image:
    if max_side is None or max_side <= 0:
        return pil
    w, h = pil.size
    m = max(w, h)
    if m <= max_side:
        return pil
    scale = max_side / float(m)
    new_size = (int(round(w * scale)), int(round(h * scale)))
    return pil.resize(new_size, Image.BILINEAR)

def _infer_core(img: Image.Image, req: InferReq):
    """
    inference._get_tf → _predict → (옵션) heatmap/overlay 생성
    """
    # 전처리 (Resize 256 → CenterCrop 224 → Normalize)
    tf = inference._get_tf(resize=256, imagesize=224)
    x_t = tf(img).unsqueeze(0).to(MODEL_DEVICE, non_blocking=True)

    # --- 순수 추론 시간 측정 (백본 + FAISS kNN) ---
    if MODEL_DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    amap = _predict(MODEL, x_t)  # (H, W) float32
    if MODEL_DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
    infer_ms = (time.perf_counter() - t0) * 1000.0
    # --------------------------------------------

    # 스코어 (예: mean)
    score = float(np.mean(amap))
    is_ng = bool(score > 0.0)

    heat_b64 = ""
    overlay_b64 = ""
    post_ms = 0.0

    if req.return_images:
        t_post0 = time.perf_counter()

        # Heatmap(그레이) - 원본 크기에서 만들고(정확), 응답용으로는 다운스케일 가능
        h = amap.astype(np.float32)
        p98 = np.percentile(h, req.clip_q * 100.0)
        lo = float(h.min()); hi = float(p98 if p98 > lo else lo + 1e-6)
        h = np.clip((h - lo) / (hi - lo), 0, 1)
        h_u8 = (h * 255.0).astype(np.uint8)
        heat_pil_full = Image.fromarray(h_u8, mode="L").resize(img.size, resample=Image.BILINEAR)

        # Overlay(원본 크기에서 생성)
        overlay_pil_full = _make_overlay(
            src_pil=img, heatmap=amap,
            clip_q=req.clip_q, alpha_min=req.alpha_min, alpha_max=req.alpha_max,
            gamma=req.gamma, blur=req.blur, normalize=True
        )

        # 응답 크기 축소(인코딩/전송 속도 개선)
        heat_pil = _resize_for_response(heat_pil_full, req.overlay_max_side)
        overlay_pil = _resize_for_response(overlay_pil_full, req.overlay_max_side)

        # 인코딩
        buf = io.BytesIO()
        if req.overlay_format.lower() == "png":
            heat_buf = io.BytesIO(); heat_pil.save(heat_buf, format="PNG")
            buf = io.BytesIO(); overlay_pil.save(buf, format="PNG")
        else:
            q = max(1, min(100, int(req.overlay_quality)))
            heat_buf = io.BytesIO(); heat_pil.save(heat_buf, format="JPEG", quality=q, optimize=False, subsampling=1)
            buf = io.BytesIO(); overlay_pil.save(buf, format="JPEG", quality=q, optimize=False, subsampling=1)

        heat_b64 = base64.b64encode(heat_buf.getvalue()).decode("ascii")
        overlay_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        post_ms = (time.perf_counter() - t_post0) * 1000.0

    return score, is_ng, heat_b64, overlay_b64, infer_ms, post_ms

# --------------------------
# 미들웨어/후크
# --------------------------
@app.on_event("startup")
def _on_start():
    logger.info("[SERVER] starting…")
    t0 = time.perf_counter()
    _load_model_once()
    _tune_faiss_threads()  # ★ 중요: FAISS 스레드 조정

    # 워밍업(옵션)
    try:
        dummy = Image.new("RGB", (224, 224), (0, 0, 0))
        _ = _infer_core(dummy, InferReq(return_images=False))
    except Exception as e:
        logger.warning(f"[WARMUP] skipped: {e}")

    logger.info(f"[SERVER] ready in {(time.perf_counter()-t0)*1000.0:.1f} ms | device={MODEL_DEVICE}")

@app.middleware("http")
async def _log_requests(request: Request, call_next):
    start = time.perf_counter()
    client = request.client.host if request.client else "unknown"
    path = request.url.path
    logger.info(f"[REQ] {client} {request.method} {path}")
    try:
        response = await call_next(request)
        ms = (time.perf_counter() - start) * 1000.0
        logger.info(f"[RES] {client} {path} {response.status_code} ({ms:.1f} ms)")
        return response
    except Exception as e:
        ms = (time.perf_counter() - start) * 1000.0
        logger.exception(f"[ERR] {client} {path} ({ms:.1f} ms) -> {e}")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

# --------------------------
# 엔드포인트
# --------------------------
@app.get("/health")
def health():
    try:
        _load_model_once()
        return {"ok": True, "device": MODEL_DEVICE}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/infer", response_model=InferRes)
def infer(req: InferReq):
    _load_model_once()
    t0 = time.perf_counter()
    try:
        img = _pil_from_req(req)
        s, ng, heat_b64, ov_b64, infer_ms, post_ms = _infer_core(img, req)
        proc_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(f"[INFER] ok score={s:.6f} NG={ng} infer={infer_ms:.1f} ms post={post_ms:.1f} ms total={proc_ms:.1f} ms")
        return InferRes(ok=True, score=s, is_ng=ng,
                        heatmap_png_b64=heat_b64, overlay_png_b64=ov_b64,
                        proc_ms=proc_ms, infer_ms=infer_ms, post_ms=post_ms)
    except Exception as e:
        proc_ms = (time.perf_counter() - t0) * 1000.0
        logger.exception(f"[INFER] failed ({proc_ms:.1f} ms) -> {e}")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

# === 신규: 멀티파트 업로드(파일 바이트) 전용 ===
@app.post("/infer_multipart", response_model=InferRes)
async def infer_multipart(
    file: UploadFile = File(...),
    clip_q: float = Form(0.98),
    alpha_min: float = Form(0.02),
    alpha_max: float = Form(0.5),
    gamma: float = Form(1.8),
    blur: float = Form(0.2),
    resize: int = Form(256),
    imagesize: int = Form(224),
    overlay_format: str = Form("jpg"),     # "png" 또는 "jpg"
    overlay_quality: int = Form(90),       # jpg 품질
    overlay_max_side: int = Form(1280),    # 응답 다운스케일(0=원본)
):
    """
    C#의 MultipartFormDataContent 업로드(B안) 전용 엔드포인트.
      - 파일 필드명: "file"
      - 나머지 파라미터는 Form으로 전달 (이름 동일)
    응답은 /infer와 동일한 InferRes 포맷.
    """
    _load_model_once()
    t0 = time.perf_counter()
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")

        # /infer와 동일한 파라미터 구조를 맞추기 위해 InferReq 구성
        req = InferReq(
            image_path=None,
            image_b64=None,
            clip_q=clip_q,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            gamma=gamma,
            blur=blur,
            return_images=True,
            overlay_format=overlay_format,
            overlay_quality=overlay_quality,
            overlay_max_side=overlay_max_side,
        )

        s, ng, heat_b64, ov_b64, infer_ms, post_ms = _infer_core(img, req)
        proc_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(f"[INFER_MP] ok score={s:.6f} NG={ng} infer={infer_ms:.1f} ms post={post_ms:.1f} ms total={proc_ms:.1f} ms")

        # response_model=InferRes 이므로 키/타입 맞춰 반환
        return {
            "ok": True,
            "score": float(s),
            "is_ng": bool(ng),
            "heatmap_png_b64": heat_b64,
            "overlay_png_b64": ov_b64,
            "proc_ms": float(proc_ms),
            "infer_ms": float(infer_ms),
            "post_ms": float(post_ms),
        }

    except Exception as e:
        proc_ms = (time.perf_counter() - t0) * 1000.0
        logger.exception(f"[INFER_MP] failed ({proc_ms:.1f} ms) -> {e}")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

@app.get("/")
def root():
    return PlainTextResponse("PatchCore FastAPI server is running.\n"
                             f"ModelDir={ENV_DIR}\nDevice={MODEL_DEVICE}\n")

if __name__ == "__main__":
    import uvicorn
    logger.info(f"[MAIN] host={HOST} port={PORT} device={DEVICE}")
    uvicorn.run(app, host=HOST, port=PORT, workers=1, log_level="info")