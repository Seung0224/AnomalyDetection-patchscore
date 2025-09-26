import cv2
import torch
import numpy as np
from PIL import Image
from PIL import Image, ImageDraw, ImageFont 

# -----------------------------
# heatmap → overlay (OpenCV 의존)
# -----------------------------
def _make_overlay(
    src_pil: Image.Image,
    heatmap: np.ndarray,
    clip_q: float = 0.8,
    alpha_min: float = 0.1,
    alpha_max: float = 0.7,
    gamma: float = 0.9,
    blur: float = 1.0,
    normalize: bool = True,
) -> Image.Image:

    # 1) heatmap normalize (percentile clipping + gamma)
    h = heatmap.astype(np.float32)
    if normalize:
        lo = np.min(h)
        hi = np.percentile(h, clip_q * 100.0)
        if hi <= lo: 
            hi = lo + 1e-6
        h = (h - lo) / (hi - lo)
        h = np.clip(h, 0.0, 1.0)
        if gamma and gamma != 1.0:
            h = np.power(h, gamma).astype(np.float32)
    else:
        # 이미 [0,1]이라고 가정
        h = np.clip(h, 0.0, 1.0)

    # 2) blur (옵션)
    if blur and blur > 0:
        # 커널: 홀수 근사
        k = max(1, int(round(blur * 2)) * 2 + 1)
        h = cv2.GaussianBlur(h, (k, k), sigmaX=0, borderType=cv2.BORDER_DEFAULT)

    # 3) heatmap 크기를 원본에 맞춤
    src = np.array(src_pil.convert("RGB"))
    H, W = src.shape[:2]
    hmap = cv2.resize(h, (W, H), interpolation=cv2.INTER_LINEAR)

    # 4) 컬러맵 (JET)
    h_u8 = (hmap * 255.0).astype(np.uint8)
    cm_bgr = cv2.applyColorMap(h_u8, cv2.COLORMAP_JET)
    cm = cv2.cvtColor(cm_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # 5) 픽셀별 알파 (alpha_min ~ alpha_max)
    a = (alpha_min + (alpha_max - alpha_min) * hmap).astype(np.float32)
    a = a[..., None]  # (H, W, 1)

    # 6) 알파 블렌딩
    src_f = src.astype(np.float32) / 255.0
    out = (1.0 - a) * src_f + a * cm
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")

def _as_numpy(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)) and x:
        x = x[0]
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    return x

def _predict(model, x_t):
    """모델에서 (anomaly_map, image_score)를 유연하게 추출해 반환.
       - 반환: (amap: (H,W) float32 ndarray, score: float or None)
       - score가 제공되지 않으면 None을 반환 (상위에서 fallback 처리)
    """
    with torch.no_grad():
        out = model.predict(x_t) if hasattr(model, "predict") else None
        if out is None:
            raise RuntimeError("PatchCore model does not expose predict().")

        amap = None
        score = None

        # 1) dict 형태 지원 (예: {'anomaly_map': ..., 'pred_score': ...})
        if isinstance(out, dict):
            if "anomaly_map" in out:
                amap = _as_numpy(out["anomaly_map"])
            if "pred_score" in out or "image_score" in out or "scores" in out:
                score = out.get("pred_score") or out.get("image_score") or out.get("scores")
                score = _as_numpy(score)

        # 2) tuple/list 형태 (예: (scores, masks) 또는 (scores, amap, ...))
        elif isinstance(out, (list, tuple)):
            # heuristic: 두 번째 항목이 heatmap/마스크, 첫 번째가 이미지 스코어
            if len(out) >= 2:
                score = _as_numpy(out[0])
                amap  = _as_numpy(out[1])

        # numpy 정리
        if isinstance(amap, (list, tuple)):
            amap = amap[0]
        if amap is None:
            raise RuntimeError("Unsupported predict() output: anomaly map not found.")

        # (H,W)로 squeeze
        amap = np.asarray(amap, dtype=np.float32)
        if amap.ndim == 3:
            amap = np.squeeze(amap)
        if amap.ndim != 2:
            raise RuntimeError(f"Unexpected anomaly map shape: {amap.shape}")

        # score scalar 뽑기 (배치/배열일 경우 0번째)
        if score is not None:
            score = np.asarray(score).reshape(-1)[0].item()

        return amap.astype(np.float32), (float(score) if score is not None else None)
    
def draw_status_frame(
    img_pil: Image.Image,
    status: str,                 # "OK" or "NG"
    score: float = None,         # e.g., 0.87
    index: int = None,           # 1-based index
    total: int = None,           # total count
    thickness: int = 10,
    marginX: int = 25,
    marginY: int = 16,
    font_size: int = 24,         # ← 글씨 크기 키움 (원하면 조절)
) -> Image.Image:
    """이미지에 상태(OK/NG) 라벨과 테두리를 그려 반환."""
    img = img_pil.convert("RGB").copy()
    w, h = img.size
    draw = ImageDraw.Draw(img)

    ok = str(status).upper() == "OK"
    color = (46, 204, 113) if ok else (231, 76, 60)  # 초록/빨강

    # 1) 외곽 테두리
    for t in range(thickness):
        draw.rectangle([t, t, w - 1 - t, h - 1 - t], outline=color)

    # 2) 라벨 텍스트 만들기  (예: "1/20 (OK) 0.873")
    parts = []
    if index is not None and total is not None and total > 0:
        parts.append(f"{index}/{total}")
    parts.append(f"({status.upper()})")
    if score is not None:
        parts.append(f"{score:.3f}")
    label = " ".join(parts)

    # 3) 폰트 및 텍스트 크기 계산
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = draw.textsize(label, font=font)

    x = w - marginX - tw
    y = marginY

    # 4) 흰 배경 박스 (가독성)
    bg_pad = 6
    draw.rectangle([x - bg_pad, y - bg_pad, x + tw + bg_pad, y + th + bg_pad], fill=(255, 255, 255))

    # 5) 텍스트: 글자색 = 테두리색, 얇은 검은 외곽선으로 가독성 보강
    try:
        draw.text((x, y), label, fill=color, font=font, stroke_width=1, stroke_fill=(0, 0, 0))
    except TypeError:
        # 구형 Pillow 대비
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            draw.text((x+dx, y+dy), label, fill=(0,0,0), font=font)
        draw.text((x, y), label, fill=color, font=font)

    return img
