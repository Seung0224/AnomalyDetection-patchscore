# model_loader.py
import os
import tempfile, shutil, hashlib
from typing import Optional, Tuple
from functools import lru_cache
import inspect, torch

from patchcore.patchcore import PatchCore
from patchcore.common import FaissNN

REQUIRED_FILES = ("patchcore_params.pkl", "nnscorer_search_index.faiss")

def validate_model_dir(model_dir: str) -> Tuple[bool, str]:
    if not model_dir or not os.path.isdir(model_dir):
        return False, f"Not a directory: {model_dir}"
    miss = [f for f in REQUIRED_FILES if not os.path.isfile(os.path.join(model_dir, f))]
    if miss:
        return False, "Missing: " + ", ".join(miss)
    return True, "OK"

# ---------- NEW: validate explicit files ----------
def validate_model_files(faiss_path: str, pkl_path: str) -> Tuple[bool, str]:
    if not faiss_path or not os.path.isfile(faiss_path):
        return False, f"FAISS file not found: {faiss_path}"
    if not pkl_path or not os.path.isfile(pkl_path):
        return False, f"PKL/PTH file not found: {pkl_path}"
    # basic ext sanity (not strict)
    faiss_ok = os.path.splitext(faiss_path)[1].lower() in {".faiss", ".index"}
    pkl_ok   = os.path.splitext(pkl_path)[1].lower() in {".pkl", ".pth", ".pt"}
    if not faiss_ok:
        return False, "FAISS must be *.faiss or *.index"
    if not pkl_ok:
        return False, "Params must be *.pkl / *.pth / *.pt"
    return True, "OK"

def _make_faiss_nn(device: str, faiss_path: str):
    params = inspect.signature(FaissNN).parameters
    faiss_nn = FaissNN(on_gpu=(device == "cuda")) if "on_gpu" in params else FaissNN()
    if hasattr(faiss_nn, "load"):
        faiss_nn.load(faiss_path)
    elif hasattr(faiss_nn, "from_file"):
        faiss_nn = faiss_nn.from_file(faiss_path)
    else:
        import faiss
        index = faiss.read_index(faiss_path)
        if hasattr(faiss_nn, "set_index"): faiss_nn.set_index(index)
        elif hasattr(faiss_nn, "index"):   faiss_nn.index = index
        else: raise RuntimeError("FaissNN index injection failed.")
    return faiss_nn

# ---------- existing cached folder loader ----------
@lru_cache(maxsize=1)
def _load_model_cached(model_dir: str, device: str) -> Tuple[PatchCore, str]:
    ok, msg = validate_model_dir(model_dir)
    if not ok: raise FileNotFoundError(msg)
    faiss_path = os.path.join(model_dir, "nnscorer_search_index.faiss")
    nn_method = _make_faiss_nn(device, faiss_path)
    needs_instance = "self" in inspect.signature(PatchCore.load_from_path).parameters
    if needs_instance:
        init_kwargs = {}
        if "device" in inspect.signature(PatchCore.__init__).parameters:
            init_kwargs["device"] = device
        model = PatchCore(**init_kwargs)
        ret = model.load_from_path(load_path=model_dir, device=device, nn_method=nn_method)
        if ret is not None:
            model = ret
    else:
        model = PatchCore.load_from_path(load_path=model_dir, device=device, nn_method=nn_method)
    model.eval()
    return model, device

def load_model(model_dir: str, device: Optional[str] = None, reset_cache: bool = False):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if reset_cache:
        _load_model_cached.cache_clear()
    return _load_model_cached(model_dir, device)

# ---------- NEW: file-based loader ----------
def _digest_for_paths(faiss_path: str, pkl_path: str) -> str:
    def stamp(p):
        try:
            st = os.stat(p)
            return f"{os.path.abspath(p)}:{st.st_mtime_ns}:{st.st_size}"
        except Exception:
            return os.path.abspath(p)
    h = hashlib.md5((stamp(faiss_path) + "|" + stamp(pkl_path)).encode()).hexdigest()
    return h

@lru_cache(maxsize=2)
def _load_model_from_files_cached(faiss_path: str, pkl_path: str, device: str):
    ok, msg = validate_model_files(faiss_path, pkl_path)
    if not ok:
        raise FileNotFoundError(msg)

    # 1) PKL 폴더 기준 ASCII-안전 캐시 루트
    pkl_dir = os.path.dirname(os.path.abspath(pkl_path))
    cache_root = _ascii_safe_cache_root(pkl_dir)

    # 2) 해시로 하위 폴더 결정
    digest = _digest_for_paths(faiss_path, pkl_path)
    load_dir = os.path.join(cache_root, digest)
    os.makedirs(load_dir, exist_ok=True)

    # 3) 표준 파일명으로 둘 다 복사/갱신
    canonical_pkl   = os.path.join(load_dir, "patchcore_params.pkl")
    canonical_faiss = os.path.join(load_dir, "nnscorer_search_index.faiss")

    def _needs_copy(dst, src):
        return (not os.path.isfile(dst) or
                os.path.getmtime(dst) < os.path.getmtime(src) or
                os.path.getsize(dst)  != os.path.getsize(src))

    if _needs_copy(canonical_pkl, pkl_path):
        shutil.copy2(pkl_path, canonical_pkl)

    if _needs_copy(canonical_faiss, faiss_path):
        shutil.copy2(faiss_path, canonical_faiss)

    # 4) FAISS 인덱스는 표준 경로에서 로드
    nn_method = _make_faiss_nn(device, canonical_faiss)

    # 5) canonical 파일들이 있는 load_dir로부터 PatchCore 로드
    needs_instance = "self" in inspect.signature(PatchCore.load_from_path).parameters
    if needs_instance:
        init_kwargs = {}
        if "device" in inspect.signature(PatchCore.__init__).parameters:
            init_kwargs["device"] = device
        model = PatchCore(**init_kwargs)
        ret = model.load_from_path(load_path=load_dir, device=device, nn_method=nn_method)
        if ret is not None:
            model = ret
    else:
        model = PatchCore.load_from_path(load_path=load_dir, device=device, nn_method=nn_method)

    model.eval()
    return model, device

def load_model_from_files(faiss_path: str, pkl_path: str, device: Optional[str] = None, reset_cache: bool = False):
    """Load PatchCore from explicit FAISS+PKL file paths (no folder assumptions)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if reset_cache:
        _load_model_from_files_cached.cache_clear()
    return _load_model_from_files_cached(os.path.abspath(faiss_path), os.path.abspath(pkl_path), device)

def _ascii_safe_cache_root(preferred_dir: str) -> str:
    """
    가능한 한 PKL이 있는 폴더 아래 ASCII 이름만 쓰는 캐시 루트를 만든다.
    실패하면 현재 작업 폴더로 폴백.
    """
    try:
        root = os.path.join(preferred_dir, "_adi_patchcore_models")
        os.makedirs(root, exist_ok=True)
        # 쓰기 테스트
        test = os.path.join(root, "._w")
        open(test, "w").close(); os.remove(test)
        return root
    except Exception:
        root = os.path.join(os.getcwd(), "adi_patchcore_models")
        os.makedirs(root, exist_ok=True)
        return root