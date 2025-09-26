import cv2
import torch
import numpy as np
from PIL import Image

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

def _predict(model, x_t):
    with torch.no_grad():
        if hasattr(model, "predict"):
            out = model.predict(x_t)
            if isinstance(out, (list, tuple)) and len(out) >= 2:
                _, masks = out[:2]
                amap = masks[0]
            else:
                raise RuntimeError("Unsupported predict() output shape for PatchCore.")
        else:
            raise RuntimeError("PatchCore model does not expose anomaly map method.")

        # tensor/list/ndarray 정리
        if isinstance(amap, (list, tuple)):
            amap = amap[0]
        if hasattr(amap, "detach"):
            amap = amap.detach().float().cpu().numpy()
        else:
            amap = np.asarray(amap, dtype=np.float32)

        # (H, W)로 맞추기
        if amap.ndim == 3:
            amap = np.squeeze(amap)
        return amap.astype(np.float32)