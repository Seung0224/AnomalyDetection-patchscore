# CableSwap.py
# -----------------------------------------------------------------------------
# Rule-based checker for cable color order (position swap) and color dominance.
# Works WITHOUT ring ROI. Designed to plug into your existing GUI quickly.
#
# - Accepts PIL.Image, NumPy array, tensor, or file path.
# - Uses HSV color thresholds to isolate blue / green / black regions.
# - Picks the largest connected component per color and compares x-centroids.
# - Reports:
#     * position swap violation (predicted left→right order vs expected order)
#     * dominance violation (one color occupies too much relative area)
# - Optional: draw annotations (centroids, bounding boxes, order text).
#
# Default expected order is (blue, green, black) → (0,1,2).
# Adjust NORMAL_ORDER below if your normal differs.
# -----------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import os
import cv2
import numpy as np
from PIL import Image

# ----------------------------
# Public constants / defaults
# ----------------------------
COLOR_NAMES = ["blue", "green", "black"]
COLOR_TO_IDX = {"blue": 0, "green": 1, "black": 2}
IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}

# Expected normal left→right order (blue=0, green=1, black=2)
NORMAL_ORDER: Tuple[int, int, int] = (0, 1, 2)

# HSV thresholds (OpenCV HSV: H∈[0,179], S,V∈[0,255])
DEFAULT_S_MIN = 35
DEFAULT_V_MIN = 20

# Dominance threshold: if one color occupies ≥ this fraction of the union area → violation
DOMINANCE_FRAC = 0.50  # 50%
# Or if one color is ≥ RATIO×(mean of others), treat as dominance too.
DOMINANCE_RATIO = 1.6

# Area sanity constraints
MIN_COLOR_AREA_FRAC = 0.001  # each color's largest component must be at least 0.1% of image area (fallback lowers this)


ImgLike = Union[str, Image.Image, np.ndarray, bytes, memoryview]


@dataclass
class CableSwapResult:
    ok: bool
    # estimated order and areas
    pred_order: Optional[Tuple[int, int, int]]
    color_areas: Dict[str, int]
    color_centroids: Dict[str, Optional[Tuple[float, float]]]
    union_area: int

    # violations
    violation_pos: bool
    violation_dominance: bool
    dominant_color: Optional[str]

    # debug / tuning
    params: Dict[str, Any]
    note: str = ""

    def any_violation(self) -> bool:
        return bool(self.violation_pos or self.violation_dominance)


# ----------------------------
# Utility: image loading
# ----------------------------
def _to_bgr(img: ImgLike) -> Optional[np.ndarray]:
    if isinstance(img, Image.Image):
        return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    if isinstance(img, str):
        if not os.path.isfile(img):
            return None
        im = Image.open(img).convert("RGB")
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    if isinstance(img, bytes) or isinstance(img, memoryview):
        from io import BytesIO
        im = Image.open(BytesIO(img)).convert("RGB")
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    if isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        # Assume RGB if looks like PIL output; try to detect
        # Heuristic: if average of first channel is closer to green than red it's OK either way;
        # we simply try RGB->BGR conversion if it "looks" RGB (common in PIL).
        # For safety, treat as RGB unless user passes already BGR.
        # We provide a flag toggle via analyze_image(..., assume_bgr=True).
        return arr
    return None


# ----------------------------
# Color masking
# ----------------------------
def _hsv_ranges(s_min=DEFAULT_S_MIN, v_min=DEFAULT_V_MIN):
    blue  = ((100, s_min, v_min), (130, 255, 255))
    green = (( 35, s_min, v_min), ( 85, 255, 255))
    # black: low value/sat – keep conservative to avoid background; tightened a bit
    black = ((  0,   0,   0), (179,  70,  90))
    return {"blue": blue, "green": green, "black": black}


def _largest_component(bin_mask: np.ndarray) -> Tuple[Optional[int], Optional[Tuple[float, float]], Optional[Tuple[int,int,int,int]]]:
    # returns (area, centroid(x,y), bbox(x,y,w,h))
    if bin_mask is None or bin_mask.size == 0:
        return None, None, None
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    if nlabels <= 1:
        return None, None, None
    # exclude background (label 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx   = int(np.argmax(areas)) + 1
    area  = int(areas[idx - 1])
    cx, cy = centroids[idx]
    x, y, w, h = stats[idx, :4]
    return area, (float(cx), float(cy)), (int(x), int(y), int(w), int(h))


def _prep_masks(bgr: np.ndarray,
                s_min=DEFAULT_S_MIN, v_min=DEFAULT_V_MIN,
                morph_open=True) -> Dict[str, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ranges = _hsv_ranges(s_min, v_min)
    out = {}
    kernel = np.ones((3,3), np.uint8)
    for name, (lo, hi) in ranges.items():
        mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        if morph_open:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        out[name] = mask
    return out


def _restrict_black_to_color_neighborhood(masks: Dict[str, np.ndarray]) -> None:
    """Black mask can be huge if background is dark. Restrict it near blue/green blobs."""
    blue = masks["blue"]
    green = masks["green"]
    black = masks["black"]

    seed = cv2.bitwise_or(blue, green)
    if cv2.countNonZero(seed) == 0:
        return  # keep original black; nothing to restrict to

    # grow seed to region around colored cables
    kernel = np.ones((9,9), np.uint8)
    region = cv2.dilate(seed, kernel, iterations=2)
    masks["black"] = cv2.bitwise_and(black, region)


# ----------------------------
# Core analysis
# ----------------------------
def analyze_image(img: ImgLike,
                  normal_order: Sequence[int] = NORMAL_ORDER,
                  s_min: int = DEFAULT_S_MIN,
                  v_min: int = DEFAULT_V_MIN,
                  dominance_frac: float = DOMINANCE_FRAC,
                  dominance_ratio: float = DOMINANCE_RATIO,
                  assume_bgr: bool = False) -> CableSwapResult:
    """
    Returns CableSwapResult with position-swap and dominance checks.
    """
    bgr = _to_bgr(img)
    if bgr is None:
        return CableSwapResult(
            ok=False, pred_order=None, color_areas={}, color_centroids={},
            union_area=0, violation_pos=False, violation_dominance=False,
            dominant_color=None, params=dict(normal_order=tuple(normal_order), s_min=s_min, v_min=v_min),
            note="failed to load image"
        )
    if not assume_bgr:
        # Heuristic: treat numpy arrays as RGB by default (typical PIL output) → convert to BGR
        bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)

    H, W = bgr.shape[:2]
    area_img = H * W

    masks = _prep_masks(bgr, s_min=s_min, v_min=v_min, morph_open=True)
    _restrict_black_to_color_neighborhood(masks)

    # Largest component per color
    areas: Dict[str, int] = {}
    cents: Dict[str, Optional[Tuple[float,float]]] = {}
    boxes: Dict[str, Optional[Tuple[int,int,int,int]]] = {}

    for cname in COLOR_NAMES:
        area, cent, bbox = _largest_component(masks[cname])
        areas[cname] = int(area) if area is not None else 0
        cents[cname] = cent
        boxes[cname] = bbox

    # If everything is tiny, relax thresholds once
    if sum(areas.values()) < area_img * 0.002:
        masks = _prep_masks(bgr, s_min=max(0, s_min-5), v_min=max(0, v_min-5), morph_open=False)
        _restrict_black_to_color_neighborhood(masks)
        for cname in COLOR_NAMES:
            area, cent, bbox = _largest_component(masks[cname])
            areas[cname] = int(area) if area is not None else 0
            cents[cname] = cent
            boxes[cname] = bbox

    # Union area (for dominance fraction)
    union = np.bitwise_or(np.bitwise_or(masks["blue"]>0, masks["green"]>0), masks["black"]>0)
    union_area = int(np.count_nonzero(union))

    # Compute predicted order by x-centroid if all three present
    found_all = all(cents[c] is not None and areas[c] >= max(1, int(area_img*MIN_COLOR_AREA_FRAC)) for c in COLOR_NAMES)
    if found_all:
        xs = [(c, cents[c][0]) for c in COLOR_NAMES]
        xs_sorted = sorted(xs, key=lambda t: t[1])
        pred_names = [c for c,_ in xs_sorted]
        pred_order = tuple(COLOR_TO_IDX[c] for c in pred_names)
        violation_pos = (tuple(normal_order) != pred_order)
    else:
        pred_order = None
        violation_pos = False  # cannot decide

    # Dominance: by union fraction and ratio vs others
    if union_area > 0:
        fracs = {c: areas[c]/union_area for c in COLOR_NAMES}
        dom_color = max(COLOR_NAMES, key=lambda c: fracs[c])
        dom_frac = fracs[dom_color]
        others = [fracs[c] for c in COLOR_NAMES if c != dom_color]
        mean_others = (others[0] + others[1]) / 2.0 if len(others)==2 else 0.0
        violation_dominance = (dom_frac >= dominance_frac) or (mean_others > 0 and (dom_frac / mean_others) >= dominance_ratio)
    else:
        dom_color = None
        violation_dominance = False

    return CableSwapResult(
        ok=True,
        pred_order=pred_order,
        color_areas=areas,
        color_centroids=cents,
        union_area=union_area,
        violation_pos=violation_pos,
        violation_dominance=violation_dominance,
        dominant_color=dom_color,
        params=dict(normal_order=tuple(normal_order), s_min=s_min, v_min=v_min,
                    dominance_frac=dominance_frac, dominance_ratio=dominance_ratio,
                    assume_bgr=assume_bgr),
        note=("all colors found" if found_all else "insufficient color components for order test")
    )


# ----------------------------
# Visualization (optional)
# ----------------------------
def draw_annotations(img: ImgLike, result: CableSwapResult,
                     thickness: int = 2, font_scale: float = 0.6) -> Image.Image:
    bgr = _to_bgr(img)
    if bgr is None:
        raise ValueError("Cannot load image for annotation.")
    bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)  # treat PIL/np RGB by default

    col_bgr = {"blue": (255, 80, 80), "green": (80, 255, 80), "black": (30, 30, 30)}
    for cname, cent in result.color_centroids.items():
        if cent is None: continue
        cx, cy = int(round(cent[0])), int(round(cent[1]))
        cv2.circle(bgr, (cx, cy), 6, col_bgr.get(cname, (255,255,255)), -1)
        cv2.putText(bgr, cname[0].upper(), (cx+6, cy-6), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, col_bgr.get(cname, (255,255,255)), thickness, cv2.LINE_AA)

    # Order banner
    if result.pred_order is not None:
        names = [IDX_TO_COLOR[i][0].upper() for i in result.pred_order]
        txt = "Predicted order: " + "→".join(names)
        cv2.putText(bgr, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240,240,240), 2, cv2.LINE_AA)

    # Violations
    y = 48
    if result.violation_pos:
        cv2.putText(bgr, "[VIOLATION] position swap", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80,20,240), 2, cv2.LINE_AA); y += 24
    if result.violation_dominance:
        cv2.putText(bgr, f"[VIOLATION] dominance: {result.dominant_color}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80,20,240), 2, cv2.LINE_AA); y += 24

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb, mode="RGB")


# ----------------------------
# Optional fused score helper
# ----------------------------
def compute_fused_score(patchcore_score: float,
                        swap_result: CableSwapResult,
                        alpha: float = 1.8) -> float:
    swap_pos = 1.0 if swap_result.violation_pos else 0.0
    swap_dom = 1.0 if swap_result.violation_dominance else 0.0
    swap_score = max(swap_pos, swap_dom)
    return max(float(patchcore_score), float(alpha) * float(swap_score))


# ----------------------------
# CLI (standalone test)
# ----------------------------
def _fmt_order(o: Optional[Tuple[int,int,int]]) -> str:
    if o is None: return "N/A"
    return "→".join(IDX_TO_COLOR[i][0].upper() for i in o)

def _fmt_areas(d: Dict[str,int]) -> str:
    return ", ".join(f"{k}:{v}" for k,v in d.items())

def _main():
    import argparse, json, glob, os
    p = argparse.ArgumentParser("CableSwap checker (ringless)")
    p.add_argument("paths", nargs="+", help="image file(s) or a folder (glob supported)")
    p.add_argument("--order", type=str, default="B,G,K", help="normal order e.g. B,G,K or G,B,K (B=blue,G=green,K=black)")
    p.add_argument("--smin", type=int, default=DEFAULT_S_MIN)
    p.add_argument("--vmin", type=int, default=DEFAULT_V_MIN)
    p.add_argument("--dom-frac", type=float, default=DOMINANCE_FRAC)
    p.add_argument("--dom-ratio", type=float, default=DOMINANCE_RATIO)
    p.add_argument("--annotate", action="store_true", help="save annotated images next to originals")
    args = p.parse_args()

    def parse_order(txt: str) -> Tuple[int,int,int]:
        mapc = {"B":0, "BLUE":0, "G":1, "GREEN":1, "K":2, "BLACK":2}
        parts = [t.strip().upper() for t in txt.split(",")]
        if len(parts) != 3 or any(p not in mapc for p in parts):
            raise SystemExit("Invalid --order. Use like B,G,K")
        return (mapc[parts[0]], mapc[parts[1]], mapc[parts[2]])

    normal_order = parse_order(args.order)

    # expand paths
    files = []
    for pat in args.paths:
        if os.path.isdir(pat):
            files += [os.path.join(pat, n) for n in os.listdir(pat)
                      if os.path.splitext(n)[1].lower() in {".png",".jpg",".jpeg",".bmp"}]
        else:
            files += glob.glob(pat)
    files = sorted(files)

    results = []
    for f in files:
        res = analyze_image(f, normal_order=normal_order, s_min=args.smin, v_min=args.vmin,
                            dominance_frac=args.dom_frac, dominance_ratio=args.dom_ratio)
        results.append({
            "path": f,
            "ok": res.ok,
            "pred_order": res.pred_order,
            "violation_pos": res.violation_pos,
            "violation_dominance": res.violation_dominance,
            "dominant_color": res.dominant_color,
            "areas": res.color_areas,
            "union_area": res.union_area,
            "note": res.note
        })
        print(f"{os.path.basename(f)} | order={_fmt_order(res.pred_order)} | "
              f"pos_swap={res.violation_pos} | dom={res.violation_dominance}({res.dominant_color}) | "
              f"areas={_fmt_areas(res.color_areas)} | note={res.note}")
        if args.annotate:
            try:
                im = draw_annotations(f, res)
                stem, ext = os.path.splitext(f)
                out = stem + "_cableswap" + ext
                im.save(out)
            except Exception:
                pass

    # also write a JSON next to the first file's folder
    if results:
        out_json = os.path.join(os.path.dirname(files[0]) if files else ".", "cableswap_results.json")
        try:
            with open(out_json, "w", encoding="utf-8") as fp:
                json.dump(results, fp, ensure_ascii=False, indent=2, default=str)
            print(f"Saved: {out_json}")
        except Exception as e:
            print(f"Failed to save JSON: {e}")

if __name__ == "__main__":
    _main()

# ----------------------------
# Geometry helper (triangular layout: Green on top, Blue left, Black right)
# ----------------------------
def check_tri_top_green(result: CableSwapResult,
                        image_shape: Tuple[int, int],
                        tol_frac_xy: Tuple[float, float] = (0.02, 0.02)) -> Optional[bool]:
    """
    Returns True if geometry is violated under the triangular layout assumption:
      - Blue is LEFT of Black (x_B + tolx < x_K)
      - Green is ABOVE both (y_G + toly < min(y_B, y_K))
    If centroids are missing, returns None (cannot decide).

    Args:
        result: output from analyze_image(...)
        image_shape: (H, W)
        tol_frac_xy: fractional tolerances (x, y) relative to (W, H)

    Example:
        res = analyze_image(img)
        viol = check_tri_top_green(res, img.shape[:2], tol_frac_xy=(0.03, 0.03))
        if viol is True: ...  # geometry violation
    """
    H, W = int(image_shape[0]), int(image_shape[1])
    if not all(result.color_centroids.get(c) is not None for c in ["blue", "green", "black"]):
        return None

    xB, yB = result.color_centroids["blue"]
    xG, yG = result.color_centroids["green"]
    xK, yK = result.color_centroids["black"]

    tolx = float(tol_frac_xy[0]) * float(W)
    toly = float(tol_frac_xy[1]) * float(H)

    cond_lr = (xB + tolx) < xK           # Blue strictly left of Black within tolerance
    cond_top = (yG + toly) < min(yB, yK) # Green noticeably above both

    return not (cond_lr and cond_top)

# ----------------------------
# Heatmap generator (no ring ROI)
# ----------------------------
def make_swap_heatmap(img: ImgLike,
                      normal_order: Sequence[int] = NORMAL_ORDER,
                      tol_frac_xy: Tuple[float, float] = (0.03, 0.03),
                      s_min: int = DEFAULT_S_MIN,
                      v_min: int = DEFAULT_V_MIN,
                      dominance_frac: float = DOMINANCE_FRAC,
                      dominance_ratio: float = DOMINANCE_RATIO,
                      alpha: float = 0.65) -> Image.Image:
    """
    Build a per-pixel 'violation' heatmap and overlay on the original image.
    Highlights regions that contradict the expected geometry (Blue<-left | Black->right | Green^top)
    and dominance overflow.

    Returns: PIL.Image (RGB) with colored heatmap overlay.
    """
    bgr = _to_bgr(img)
    if bgr is None:
        raise ValueError("Failed to load image")
    bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)  # assume input is RGB-like

    H, W = bgr.shape[:2]
    X = np.tile(np.arange(W, dtype=np.float32)[None, :], (H, 1))
    Y = np.tile(np.arange(H, dtype=np.float32)[:, None], (1, W))

    # masks and stats
    masks = _prep_masks(bgr, s_min=s_min, v_min=v_min, morph_open=True)
    _restrict_black_to_color_neighborhood(masks)

    areas = {}
    cents = {}
    for cname in COLOR_NAMES:
        a, c, _ = _largest_component(masks[cname])
        areas[cname] = int(a) if a is not None else 0
        cents[cname] = c

    # mid line between Blue and Black centroids (fallback to image center)
    xB = cents["blue"][0] if cents["blue"] is not None else None
    xK = cents["black"][0] if cents["black"] is not None else None
    if xB is not None and xK is not None:
        midx = 0.5 * (xB + xK)
    else:
        midx = W * 0.5

    yB = cents["blue"][1] if cents["blue"] is not None else None
    yK = cents["black"][1] if cents["black"] is not None else None
    if yB is not None and yK is not None:
        topy = min(yB, yK)
    else:
        topy = H * 0.5

    tolx = float(tol_frac_xy[0]) * float(W)
    toly = float(tol_frac_xy[1]) * float(H)

    # penalty maps (0..1)
    heat = np.zeros((H, W), dtype=np.float32)

    # Blue should be LEFT of (midx + tolx)
    if np.any(masks["blue"]):
        wrong_blue = (X > (midx + tolx)).astype(np.float32)
        # distance-based penalty scaled by 0.2*W
        penB = np.clip((X - (midx + tolx)) / (0.2 * W), 0.0, 1.0)
        heat = np.maximum(heat, penB * (masks["blue"] > 0))

    # Black should be RIGHT of (midx - tolx)
    if np.any(masks["black"]):
        penK = np.clip(((midx - tolx) - X) / (0.2 * W), 0.0, 1.0)
        heat = np.maximum(heat, penK * (masks["black"] > 0))

    # Green should be ABOVE (topy - toly)
    if np.any(masks["green"]):
        penG = np.clip((Y - (topy - toly)) / (0.25 * H), 0.0, 1.0)
        heat = np.maximum(heat, penG * (masks["green"] > 0))

    # Dominance overflow: add uniform boost over dominant color area
    union = (masks["blue"] > 0) | (masks["green"] > 0) | (masks["black"] > 0)
    union_area = int(np.count_nonzero(union))
    if union_area > 0:
        fracs = {c: areas[c] / union_area for c in COLOR_NAMES}
        dom_c = max(COLOR_NAMES, key=lambda c: fracs[c])
        dom_frac = fracs[dom_c]
        others = [fracs[c] for c in COLOR_NAMES if c != dom_c]
        mean_other = (others[0] + others[1]) / 2.0 if len(others) == 2 else 0.0
        dom_violation = (dom_frac >= dominance_frac) or (mean_other > 0 and (dom_frac/mean_other) >= dominance_ratio)
        if dom_violation:
            # boost is how much it exceeds the threshold (0..1)
            excess = max(0.0, float(dom_frac - dominance_frac)) / max(1e-6, (1.0 - dominance_frac))
            boost = np.clip(0.3 + 0.7 * excess, 0.3, 1.0)  # 0.3..1.0
            heat = np.maximum(heat, boost * (masks[dom_c] > 0).astype(np.float32))

    heat = np.clip(heat, 0.0, 1.0)
    heat_u8 = (heat * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)  # BGR
    over = cv2.addWeighted(bgr, 1.0, colored, float(alpha), 0)

    rgb = cv2.cvtColor(over, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb, mode="RGB")
