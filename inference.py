# inference.py
import os
from typing import Optional, Union
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from overlay_utils import _predict, _make_overlay
from functools import lru_cache

# -----------------------------------------------------------------------------
# [전처리 파이프라인 설명]
#
# PatchCore에서 사용하는 백본(ResNet, ViT 등)은 "ImageNet" 데이터셋으로
# 사전학습(pretrain)된 모델입니다. 따라서 추론에 넣는 입력 이미지는
# 훈련 시 모델이 보던 입력과 동일한 조건으로 맞춰야 성능이 제대로 나옵니다.
#
# 이 과정을 PyTorch에서는 transforms.Compose([...]) 로 정의합니다.
#
# 1) Resize(256)
#    - 이미지를 짧은 변 기준으로 256 픽셀 크기로 리사이즈합니다.
#    - 이유: 모델은 훈련 때 비슷한 크기(예: 224x224 crop)로 학습되었습니다.
#            원본이 너무 크거나 작으면 입력 통계가 달라져서 성능이 저하됩니다.
#    - 즉, 입력 크기를 일단 통일하기 위한 과정입니다.
#
# 2) CenterCrop(224)
#    - 리사이즈된 이미지 중앙에서 224x224 영역만 잘라냅니다.
#    - 이유: 대부분의 ImageNet 프리트레인 모델은 224x224 크기로 학습되었습니다.
#            입력 크기가 다르면 파라미터 shape가 맞지 않거나 성능이 떨어집니다.
#    - 즉, 백본이 "딱 맞게" 기대하는 크기로 잘라주는 단계입니다.
#
# 3) ToTensor()
#    - PIL.Image (H×W×C, 값 범위 0~255, dtype=uint8)를
#      torch.Tensor (C×H×W, 값 범위 0~1, dtype=float32)로 변환합니다.
#    - 이유:
#        * PyTorch는 Tensor를 입력으로 받습니다.
#        * CNN 백본은 channel-first (C×H×W) 포맷을 기대합니다.
#        * 픽셀 값을 float32 (0~1)로 맞춰야 학습 분포와 일치합니다.
#
# 4) Normalize(mean, std)
#    - 각 채널별로 x = (x - mean) / std 로 정규화합니다.
#    - mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
#    - 이유: 이 값들은 ImageNet 데이터셋에서 계산된 RGB 채널 평균/표준편차입니다.
#            ImageNet으로 학습된 모델은 학습 중에도 이렇게 정규화된 입력만 봤습니다.
#            추론 시에도 똑같이 맞춰줘야 성능이 유지됩니다.
#
# [직관적인 비유]
# - Resize + CenterCrop : "모델이 보는 사진은 항상 여권 사진처럼 224×224로 맞춰라"
# - ToTensor            : "사진을 딥러닝이 읽을 수 있는 숫자 행렬로 바꿔라"
# - Normalize           : "모델이 ImageNet에서 본 색감/밝기 기준에 맞게 조정해라"
#
# [예시: 픽셀 (200,50,50) 빨간색]
# - 원본 uint8   : [200, 50, 50] (0~255 범위)
# - ToTensor()   : [0.784, 0.196, 0.196] (0~1 범위)
# - Normalize()  : [ 1.31, -1.16, -0.93 ] (ImageNet 통계 맞춘 분포, 보통 -2~+2)
# - 이렇게 “Normalize”까지 거치면, 값이 -2 ~ +2 정도의 분포로 맞춰져요.
# - 딥러닝 백본(ResNet, ViT 등)은 학습할 때 항상 이런 입력을 보았기 때문에, 추론 시에도 똑같이 맞춰줘야 성능이 나옵니다.
# -----------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _get_tf(resize: int, imagesize: int):
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(imagesize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

ImgLike = Union[str, Image.Image, np.ndarray, torch.Tensor, bytes, memoryview]

def anomaly_detect(img: ImgLike, model, device: str, **kwargs,) -> Optional[Image.Image]:
    resize    = kwargs.pop("resize", 256)
    imagesize = kwargs.pop("imagesize", 224)
    clip_q    = kwargs.pop("clip_q", 0.98)
    alpha_min = kwargs.pop("alpha_min", 0.02)
    alpha_max = kwargs.pop("alpha_max", 0.5)
    gamma     = kwargs.pop("gamma", 1.8)
    blur      = kwargs.pop("blur", 0.2)
    
    if kwargs:
        raise TypeError(f"Unknown options: {list(kwargs.keys())}")

    # 1) 입력 통일: PIL.Image 확보
    x_pil = _convert_imageformat(img)
    if x_pil is None:
        return None
    
    # 2) 전처리 → 텐서
    tf = _get_tf(resize, imagesize)
    x_t = tf(x_pil).unsqueeze(0).to(device, non_blocking=True)

    # 3) Predict
    amap = _predict(model, x_t)     # (H, W) float32

    # 4) Overlay
    overlay = _make_overlay(
    src_pil=x_pil, heatmap=amap,
    clip_q=clip_q, alpha_min=alpha_min, alpha_max=alpha_max,
    gamma=gamma, blur=blur
    )
    return overlay

def _convert_imageformat(img: ImgLike) -> Optional[Image.Image]:
    """여러 입력 타입을 PIL.Image(RGB)로 통일."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, str):
        if not os.path.isfile(img):
            return None
        return Image.open(img).convert("RGB")
    if isinstance(img, bytes) or isinstance(img, memoryview):
        from io import BytesIO
        return Image.open(BytesIO(img)).convert("RGB")
    if isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 2:  # grayscale
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 4:  # RGBA → RGB
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")
    if torch.is_tensor(img):
        t = img
        # 허용: HWC uint8 or CHW float/uint8
        if t.ndim == 3:
            if t.dtype == torch.uint8 and t.shape[-1] in (1,3,4):  # HWC
                arr = t.cpu().numpy()
                return _convert_imageformat(arr)
            # CHW
            if t.shape[0] in (1,3,4):
                c,h,w = t.shape
                if t.dtype.is_floating_point:
                    t = (t.clamp(0,1) * 255).to(torch.uint8)
                arr = t.permute(1,2,0).cpu().numpy()
                return _convert_imageformat(arr)
    return None