# export_wrn50_l2l3.py
import torch
import torch.nn as nn
import torchvision.models as tvm
from pathlib import Path

# 1) WRN50-2 백본에서 layer2, layer3를 뽑아내는 래퍼
class WRN50_L23(nn.Module):
    def __init__(self, weights="IMAGENET1K_V1"):
        super().__init__()
        # torchvision 0.13+ 표준 API
        try:
            w = getattr(tvm, "Wide_ResNet50_2_Weights")[weights]
            m = tvm.wide_resnet50_2(weights=w)
        except Exception:
            # 구버전 호환(Deprecated)
            m = tvm.wide_resnet50_2(pretrained=True)

        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.relu = m.relu
        self.maxpool = m.maxpool
        self.layer1 = m.layer1
        self.layer2 = m.layer2  # 28x28
        self.layer3 = m.layer3  # 14x14

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        l2 = self.layer2(x)          # [B, 512, 28, 28]
        l3 = self.layer3(l2)         # [B, 1024, 14, 14]
        return l2, l3

if __name__ == "__main__":
    # artifacts 폴더 만들기
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = str(out_dir / "wrn50_l2l3.onnx")

    device = "cpu"
    model = WRN50_L23().to(device).eval()

    dummy = torch.randn(1, 3, 224, 224, device=device)

    # 2) ONNX export
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"],
        output_names=["layer2", "layer3"],
        opset_version=12,
        dynamic_axes={"input": {0: "batch"}, "layer2": {0: "batch"}, "layer3": {0: "batch"}},
    )
    print(f"[OK] Exported: {onnx_path}")

    # 3) 빠른 검증
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    outs = sess.run(None, {"input": dummy.numpy()})
    l2, l3 = outs
    print("layer2 shape:", l2.shape)  # (1, 512, 28, 28)
    print("layer3 shape:", l3.shape)  # (1, 1024, 14, 14)