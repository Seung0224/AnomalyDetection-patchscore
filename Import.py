import importlib.util, sys
mods = ["patchcore", "faiss", "timm"]
for m in mods:
    ok = importlib.util.find_spec(m) is not None
    print(f"[check] {m:9s}:", "OK" if ok else "MISSING")
import torch
print("[torch] CUDA available:", torch.cuda.is_available())