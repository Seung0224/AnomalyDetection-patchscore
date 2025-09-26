# openvino_backbone.py  (네이티브 OpenVINO + ORT-CPU 폴백, print 로그 확장/수정)
from __future__ import annotations
import os, sys, time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

# ==== 로드 비콘 ====
print(f"[OVFILE] loaded: {__file__}")

# ---------------- (선택) ONNX 내보내기: 원본 유지 ----------------
class _WRN50_L2L3(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            weights = models.Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.wide_resnet50_2(weights=weights)
        except Exception:
            net = models.wide_resnet50_2(pretrained=pretrained)
        net.eval()
        self.conv1=net.conv1; self.bn1=net.bn1; self.relu=net.relu; self.maxpool=net.maxpool
        self.layer1=net.layer1; self.layer2=net.layer2; self.layer3=net.layer3
    def forward(self, x):
        x=self.conv1(x); x=self.bn1(x); x=self.relu(x); x=self.maxpool(x)
        x=self.layer1(x); x=self.layer2(x); f2=x; x=self.layer3(x); f3=x
        return (f2,f3)

def export_wrn50_l2l3_onnx(out_path: str = "wrn50_l2l3.onnx", opset: int = 12, dynamic: bool = True):
    model = _WRN50_L2L3(pretrained=True).eval()
    dummy = torch.zeros(1,3,224,224,dtype=torch.float32)
    input_names=["input"]; output_names=["layer2","layer3"]
    dynamic_axes={"input":{0:"N"},"layer2":{0:"N"},"layer3":{0:"N"}} if dynamic else None
    torch.onnx.export(model, dummy, out_path, input_names=input_names, output_names=output_names,
                      dynamic_axes=dynamic_axes, opset_version=opset, do_constant_folding=True)
    print(f"[EXPORT] wrote ONNX to {out_path}")
    return out_path

# ---------------- OpenVINO 네이티브 추론기 ----------------
# (Deprecation 경고 최소화를 위해 openvino 우선)
try:
    from openvino import Core, get_version as _ov_version
except Exception:
    try:
        from openvino.runtime import Core, get_version as _ov_version
    except Exception:
        Core = None
        def _ov_version(): return "not-available"

def _shape(x) -> Tuple[int, ...]:
    try: return tuple(x.shape)
    except: return ()

def _ms(t0, t1): return (t1 - t0) * 1000.0

class OVNativeExtractor:
    """전처리 완료(NCHW, float32, normalize) 텐서를 입력으로 가정"""
    def __init__(self, onnx_path: str):
        if Core is None:
            raise RuntimeError("OpenVINO runtime not available")

        self.core = Core()
        # 최초 컴파일 가속 캐시
        self.core.set_property({"CACHE_DIR": "ov_cache"})

        self.model = self.core.read_model(onnx_path)

        # ⚠️ AFFINITY 제거: 최신 CPU 플러그인이 지원하지 않음
        props = {
            "PERFORMANCE_HINT": "LATENCY",
            "NUM_STREAMS": "1",
            "INFERENCE_NUM_THREADS": 8,
            # "AFFINITY": "CORE",  # <-- 제거
        }

        # 컴파일 시도 → 실패하면 키를 하나씩 제거하며 재시도
        self.compiled = self._compile_with_fallback(props)

        self.in_port = self.compiled.inputs[0]
        self.out0 = self.compiled.outputs[0]
        self.out1 = self.compiled.outputs[1]
        self.used_openvino = "native"
        self.providers_used = ["OpenVINO(Native)-CPU"]

        print(f"[OV][INIT] OpenVINO v{_ov_version()} | props={props}")
        print(f"[OV][INIT] input={self.in_port.get_partial_shape()}, "
              f"out0={self.out0.get_partial_shape()}, out1={self.out1.get_partial_shape()}")

        # 가벼운 워밍업
        try:
            req = self.compiled.create_infer_request()
            dummy = np.zeros((1,3,224,224), dtype=np.float32)
            req.infer({ self.in_port: dummy })
            print("[OV][INIT] warmup: ok")
        except Exception as e:
            print(f"[OV][INIT] warmup: failed ({e})")

    def _compile_with_fallback(self, props: dict):
        # 1차 시도
        try:
            return self.core.compile_model(self.model, "CPU", props)
        except Exception as e:
            print(f"[OV][INIT] compile failed with props {props} -> {e}")

        # 키를 하나씩 제거하며 재시도
        keys = list(props.keys())
        for k in keys:
            trial = {kk: vv for kk, vv in props.items() if kk != k}
            try:
                print(f"[OV][INIT] retry compile without {k}: {trial}")
                return self.core.compile_model(self.model, "CPU", trial)
            except Exception as e2:
                print(f"[OV][INIT] retry without {k} failed -> {e2}")

        # 최종: 기본 설정
        print("[OV][INIT] final retry with default props")
        return self.core.compile_model(self.model, "CPU")

    def __call__(self, img_tensor) -> Dict[str, torch.Tensor]:
        t_start = time.perf_counter()
        # --- 입력 처리
        if torch.is_tensor(img_tensor):
            if img_tensor.dtype != torch.float32:
                img_tensor = img_tensor.float()
            if img_tensor.ndim == 3:  # C,H,W
                img_tensor = img_tensor.unsqueeze(0)
            if img_tensor.ndim != 4 or img_tensor.shape[1] != 3:
                raise ValueError(f"[OV] Expected NCHW with C=3, got {tuple(img_tensor.shape)}")
            np_in = img_tensor.detach().cpu().numpy()
        else:
            np_in = np.asarray(img_tensor, dtype=np.float32)
            if np_in.ndim == 3:
                np_in = np_in[None, ...]
            if np_in.ndim != 4 or np_in.shape[1] != 3:
                raise ValueError(f"[OV] Expected NCHW with C=3, got {_shape(np_in)}")
        t_prep = time.perf_counter()

        # --- 추론
        req = self.compiled.create_infer_request()
        req.infer({ self.in_port: np_in })
        t_infer = time.perf_counter()

        y2 = req.get_output_tensor(0).data
        y3 = req.get_output_tensor(1).data
        out = {
            "layer2": torch.from_numpy(np.ascontiguousarray(y2)),
            "layer3": torch.from_numpy(np.ascontiguousarray(y3)),
        }
        t_wrap = time.perf_counter()

        print(f"[OV_CALL] in={_shape(np_in)} y2={_shape(y2)} y3={_shape(y3)} "
              f"| prep={_ms(t_start,t_prep):.2f}ms infer={_ms(t_prep,t_infer):.2f}ms "
              f"wrap={_ms(t_infer,t_wrap):.2f}ms total={_ms(t_start,t_wrap):.2f}ms")
        return out

# ---------------- ORT-CPU 폴백 ----------------
def _ort_version():
    try:
        import onnxruntime as ort
        return ort.__version__
    except Exception:
        return "not-available"

class ORTCPUExtractor:
    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # 필요시 스레드 튜닝: so.intra_op_num_threads = 8; so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=['CPUExecutionProvider'])
        self.input_name = self.sess.get_inputs()[0].name
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.used_openvino = "cpu"
        self.providers_used = self.sess.get_providers()
        print(f"[ORT][INIT] onnxruntime v{_ort_version()} | providers={self.providers_used}")
        try:
            dummy = np.zeros((1,3,224,224), dtype=np.float32)
            _ = self.sess.run(self.out_names, {self.input_name: dummy})
            print("[ORT][INIT] warmup: ok")
        except Exception as e:
            print(f"[ORT][INIT] warmup: failed ({e})")

    def __call__(self, img_tensor) -> Dict[str, torch.Tensor]:
        t_start = time.perf_counter()
        if torch.is_tensor(img_tensor):
            if img_tensor.dtype != torch.float32:
                img_tensor = img_tensor.float()
            if img_tensor.ndim == 3:
                img_tensor = img_tensor.unsqueeze(0)
            if img_tensor.ndim != 4 or img_tensor.shape[1] != 3:
                raise ValueError(f"[ORT] Expected NCHW with C=3, got {tuple(img_tensor.shape)}")
            np_in = img_tensor.detach().cpu().numpy()
        else:
            np_in = np.asarray(img_tensor, dtype=np.float32)
            if np_in.ndim == 3:
                np_in = np_in[None, ...]
            if np_in.ndim != 4 or np_in.shape[1] != 3:
                raise ValueError(f"[ORT] Expected NCHW with C=3, got {_shape(np_in)}")
        t_prep = time.perf_counter()

        y2, y3 = self.sess.run(self.out_names, {self.input_name: np_in})
        t_infer = time.perf_counter()

        out = {
            "layer2": torch.from_numpy(np.ascontiguousarray(y2)),
            "layer3": torch.from_numpy(np.ascontiguousarray(y3)),
        }
        t_wrap = time.perf_counter()

        print(f"[ORT_CALL] in={_shape(np_in)} y2={_shape(y2)} y3={_shape(y3)} "
              f"| prep={_ms(t_start,t_prep):.2f}ms infer={_ms(t_prep,t_infer):.2f}ms "
              f"wrap={_ms(t_infer,t_wrap):.2f}ms total={_ms(t_start,t_wrap):.2f}ms")
        return out

# ---------------- 팩토리(싱글턴) ----------------
_ov_singleton = None

def get_openvino_extractor(onnx_path: str = "wrn50_l2l3.onnx"):
    if os.getenv("USE_OPENVINO", "1").lower() not in {"1","true","yes"}:
        print("[Factory] USE_OPENVINO=off -> None (PyTorch 사용)")
        return None

    global _ov_singleton
    if _ov_singleton is not None:
        return _ov_singleton
    if not os.path.isfile(onnx_path):
        print(f"[Factory] ONNX not found: {onnx_path}")
        return None

    # 1) OpenVINO 시도
    try:
        _ov_singleton = OVNativeExtractor(onnx_path)
        print("[Factory] Backend=OpenVINO Native CPU")
        return _ov_singleton
    except Exception:
        import traceback
        print("[Factory] OpenVINO init failed:")
        traceback.print_exc()

    # 2) ORT-CPU 폴백
    try:
        _ov_singleton = ORTCPUExtractor(onnx_path)
        print("[Factory] Backend=ORT CPU Fallback")
        return _ov_singleton
    except Exception:
        import traceback
        print("[Factory] ORT init failed:")
        traceback.print_exc()
        return None

# ---------------- CLI 테스트 ----------------
def _main():
    import argparse
    from PIL import Image
    p = argparse.ArgumentParser()
    p.add_argument("--export", type=str, default="")
    p.add_argument("--test", type=str, default="")
    p.add_argument("--onnx", type=str, default="wrn50_l2l3.onnx")
    args = p.parse_args()

    if args.export:
        export_wrn50_l2l3_onnx(args.export)

    if args.test:
        tfm = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        x = tfm(Image.open(args.test).convert("RGB")).unsqueeze(0).contiguous()
        print(f"[CLI] test input shape={tuple(x.shape)}, dtype={x.dtype}")

        ov = get_openvino_extractor(args.onnx if args.onnx else (args.export or "wrn50_l2l3.onnx"))
        if ov is None:
            print("[CLI] backend unavailable (check USE_OPENVINO / ONNX path)")
            sys.exit(1)

        out = ov(x)
        print(f"[CLI] output shapes: {{k: tuple(v.shape) for k,v in out.items()}}")
        print(f"[CLI] providers={getattr(ov, 'providers_used', [])}, mode={getattr(ov, 'used_openvino', None)}")

if __name__ == "__main__":
    _main()