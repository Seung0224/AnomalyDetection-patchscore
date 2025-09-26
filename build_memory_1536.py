import os, glob, json
import numpy as np
from PIL import Image
import onnxruntime as rt
import torchvision.transforms as T
import torch

# ====== 경로 설정 ======
ONNX_PATH   = r"D:\ADI\wrn50_l2l3.onnx"      # layer2/layer3 내는 ONNX
GOOD_DIR    = r"D:\ADI\dataset\cable\train\good" # <-- 여기에 Good 이미지 폴더 경로 지정
OUT_DIR     = r"D:\ADI\patchcore\models"
MEM_OUT     = os.path.join(OUT_DIR, "memory_1536.npy")
SETTINGS_OUT= os.path.join(OUT_DIR, "settings_1536.json")

# ====== 전처리 ======
RESIZE = 256
CROP   = 224
MEAN = [0.485,0.456,0.406]
STD  = [0.229,0.224,0.225]

pre = T.Compose([
    T.Resize(RESIZE),
    T.CenterCrop(CROP),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

def l2norm(x, axis=-1, eps=1e-12):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)

def to_chw(arr, C,H,W):
    return arr.reshape(1,C,H,W)[0]  # [C,H,W]

def bilinear_resize_chw(chw, out_h, out_w):
    # torch로 간단 구현(정확/빠름)
    t = torch.from_numpy(chw).unsqueeze(0)   # [1,C,H,W]
    t2 = torch.nn.functional.interpolate(t, size=(out_h,out_w), mode="bilinear", align_corners=False)
    return t2.squeeze(0).numpy()             # [C,out_h,out_w]

def extract_1536(sess, img_pil):
    x = pre(img_pil).unsqueeze(0).numpy().astype("float32")     # [1,3,224,224]
    outs = sess.run(None, {"input": x})
    names = [o.name for o in sess.get_outputs()]
    d = {n:v for n,v in zip(names, outs)}
    l2 = to_chw(d["layer2"], 512,28,28)      # [512,28,28]
    l3 = to_chw(d["layer3"],1024,14,14)      # [1024,14,14]
    l2u = bilinear_resize_chw(l2, 14,14)     # [512,14,14]
    feat = np.concatenate([l2u, l3], axis=0) # [1536,14,14]
    feat = np.transpose(feat, (1,2,0)).reshape(-1, 1536)  # [196,1536]
    feat = l2norm(feat, axis=1).astype("float32")         # 위치별 L2 정규화
    return feat                                           # [196,1536]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sess = rt.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

    paths = []
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff"):
        paths += glob.glob(os.path.join(GOOD_DIR, ext))
    paths.sort()
    if not paths:
        raise FileNotFoundError(f"No images in {GOOD_DIR}")

    bank = []
    for i,p in enumerate(paths,1):
        try:
            img = Image.open(p).convert("RGB")
            feat = extract_1536(sess, img)     # [196,1536]
            bank.append(feat)
        except Exception as e:
            print(f"[WARN] skip {p}: {e}")
        if i%20==0: print(f"  processed {i}/{len(paths)}")

    mem = np.concatenate(bank, axis=0)        # [N,1536]
    # (선택) 간단한 다운샘플(코어셋 대신 균등 샘플): 기존 크기와 비슷하게 맞추고 싶다면 사용
    # target = 5000
    # if mem.shape[0] > target:
    #     idx = np.linspace(0, mem.shape[0]-1, target).astype(int)
    #     mem = mem[idx]

    np.save(MEM_OUT, mem.astype("float32"))
    print(f"[OK] saved {MEM_OUT}, shape={mem.shape}")

    cfg = {
        "resize": RESIZE,
        "input_size": CROP,
        "normalize_mean": MEAN,
        "normalize_std":  STD,
        "embedding_norm": "l2",
        "distance": "l2",
        "k": 1,
        "threshold": 0.0,
        "use_projection": False,
        "embedding_dim": 1536,
        "feature_map": {"H":14,"W":14}
    }
    with open(SETTINGS_OUT, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"[OK] saved {SETTINGS_OUT}")

if __name__ == "__main__":
    main()