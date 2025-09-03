# AnomalyDetection-PatchCore

**Good-only 기반 이미지 이상 탐지 CLI + PatchCore 오버레이 연동**
Windows 환경에서 **정상(양품) 데이터만으로 학습**하고, **코사인 거리 임계값**으로 이상을 판정하며, 필요 시 **PatchCore** 결과를 **트립틱/히트맵 오버레이**로 확인할 수 있는 **파이썬 기반 도구**입니다.

---

<img width="1920" height="960" alt="image" src="https://github.com/user-attachments/assets/9f89713b-ee15-4cc6-8860-e6c7a6365cf9" />
<img width="1920" height="963" alt="image" src="https://github.com/user-attachments/assets/dfc8fb44-2d9a-4cd4-98e8-02a2791ef501" />

## 📦 프로젝트 개요

* **플랫폼:** Python (Windows 권장)
* **목적:** 정상 데이터만으로 빠르게 이상 탐지 파이프라인을 구축하고 결과를 재현 가능한 CLI로 확인
* **대상 도메인:** 범용 이미지(예: 케이블, 전자부품, 외관 검사 등)
* **PatchCore 연동:** 트립틱(원본·마스크·히트) 저장 및 **원본 위 히트맵 오버레이** 생성 지원

---

## ✅ 주요 기능

### 🧠 Good-only 학습 & 추론

* **ResNet18**(ImageNet) 임베딩 → **L2 정규화** → **센트로이드(평균벡터)** → **상위 퍼센타일(예: 99.5\~99.8%) 임계값**
* 점수 = `1 − dot(f, centroid)` (작을수록 정상) / 임계값 초과 시 **NotGood**
* 학습 결과를 `normal_embed.json`으로 저장하고 즉시 재사용

### 🔥 PatchCore 결과 연동(선택)

* 평가 시 **트립틱(원본|마스크|heat)** 이미지 자동 저장
* 트립틱으로부터 **원본 위 heat 오버레이** 이미지 생성
* 오버레이 **튜닝 파라미터** 제공: `--clip_q`, `--alpha_min/max`, `--gamma`, `--blur`

### ⚙️ 의존성

* **PyTorch + torchvision 필수** (CPU/GPU 모두 지원)
* Pillow, NumPy, (선택) OpenCV, matplotlib, tqdm, rich 등

---

## 🧰 사용 방법 (CLI)

> 아래는 **예시 스크립트 이름**입니다. 실제 파일명/경로는 프로젝트 구조에 맞게 조정하세요.

### 1) Good-only 학습

```bash
# 정상(양품) 이미지 폴더 기준으로 임베딩 학습
python scripts/train_good_only.py \
  --good_dir data/good \
  --out models/normal_embed.json \
  --percentile 99.5
```

### 2) 단일/배치 추론

```bash
# 단일 이미지 추론
python scripts/infer_image.py \
  --image path/to/test.jpg \
  --embed models/normal_embed.json

# 폴더 배치 추론 (결과 CSV 저장 예시)
python scripts/batch_infer.py \
  --images path/to/test_folder \
  --embed models/normal_embed.json \
  --out out/infer_results.csv
```

### 3) PatchCore 트립틱/오버레이 (선택)

아래 예시는 PatchCore 평가 결과를 **트립틱**으로 저장하고, 이를 이용해 **히트맵 오버레이**를 생성합니다.

**모델/아티팩트**

```
D:\patchcore\models\mvtec_cable\
  ├─ patchcore_params.pkl
  └─ nnscorer_search_index.faiss
```

**필수 스크립트**

* `tools\run_eval_save_triptych_sigfix.py` → 평가 시 **triptych(원본|마스크|heat)** 저장
* `tools\make_overlay_from_triptych.py` → **triptych → 원본 위 heat 오버레이** 생성

**출력 폴더**

```
triptych: D:\patchcore\out\triptych\<불량유형>\*.png
overlay:  D:\patchcore\out\overlay_from_triptych\<불량유형>\*_heatoverlay.png
```

**자주 쓰는 실행 예**

평가 + triptych 저장

```powershell
$env:TRIPTYCH_OUT_ROOT="D:\patchcore\out\triptych"; $env:PYTHONPATH="src"
python tools\run_eval_save_triptych_sigfix.py --gpu 0 --seed 0 --save_segmentation_images D:\patchcore\out\mvtec_eval patch_core_loader -p D:\patchcore\models\mvtec_cable dataset --resize 256 --imagesize 224 -d cable mvtec D:\patchcore\data\mvtec
```

triptych → overlay

```powershell
python tools\make_overlay_from_triptych.py --root "D:\patchcore\out\triptych" --glob "**\*.png" --out "D:\patchcore\out\overlay_from_triptych" --alpha_mode value --alpha_min 0.1 --alpha_max 0.7 --clip_q 0.8 --gamma 0.9 --blur 1
```

**튜닝 포인트** (강조/노이즈 조절)

* `--clip_q`: 히트맵 상위분위 클리핑(강조 밸런스)
* `--alpha_min / --alpha_max`: 오버레이 투명도 범위
* `--gamma`: 감마 보정(시각적 대비)
* `--blur`: 블러 강도(노이즈 억제)

---

## 🧪 내부 로직 (요약)

1. **전처리:** `Resize(256/224) → CenterCrop(224) → ToTensor → ImageNet mean/std`
2. **백본:** `ResNet18` 마지막 FC를 Identity로 교체, `model.eval()`
3. **학습:** 정상 이미지 임베딩 L2 정규화 → 평균(센트로이드) → 재정규화
   임계값 = 정상셋 \*\*코사인 거리(1 − dot)\*\*의 **상위 퍼센타일**(예: 99.5%\~99.8%)
4. **추론:** 동일 전처리/임베딩 → 점수 산출 → `score > threshold` 이면 **NotGood**, 아니면 **Good**
5. **아티팩트:** `normal_embed.json`에 `backbone`, `input_size=224`, `mean/std`, `centroid`, `threshold` 저장

---

## 🔧 개발 환경 및 라이브러리

| 구성 요소          | 내용                                                                              |
| -------------- | ------------------------------------------------------------------------------- |
| 언어/런타임         | Python 3.9\~3.11                                                                |
| DL 백엔드         | PyTorch, torchvision                                                            |
| 이미지 처리         | Pillow, OpenCV (선택)                                                             |
| 시각화/유틸         | matplotlib, tqdm, rich (선택)                                                     |
| PatchCore 스크립트 | `tools/run_eval_save_triptych_sigfix.py`, `tools/make_overlay_from_triptych.py` |

---

## 📁 프로젝트 구조 (일부)

```
AnomalyDetection/
 ├── scripts/
 │    ├── train_good_only.py        # 정상 임베딩 학습 (예시)
 │    ├── infer_image.py            # 단일 이미지 추론 (예시)
 │    └── batch_infer.py            # 폴더 배치 추론 (예시)
 ├── tools/
 │    ├── run_eval_save_triptych_sigfix.py
 │    └── make_overlay_from_triptych.py
 ├── src/                            # 모델/유틸
 ├── models/                         # 학습/임베딩/파라미터 저장
 │    └── normal_embed.json
 ├── data/                           # 입력 이미지 (Good/테스트)
 ├── out/
 │    ├── triptych/                  # PatchCore 트립틱 출력
 │    └── overlay_from_triptych/     # 히트맵 오버레이 결과
 ├── requirements.txt
 └── README.md
```

> ※ `scripts/*` 파일명은 실제 구현에 맞게 변경하세요. GUI(Tkinter) 파일은 포함하지 않습니다.

---

## 📦 릴리즈 노트

| 날짜         | 버전     | 주요 변경 내용                                                 |
| ---------- | ------ | -------------------------------------------------------- |
| 2025-09-01 | v0.9.0 | Good-only 파이프라인 초기 구현 (CLI 베이스)                          |
| 2025-09-02 | v0.9.1 | 퍼센타일 임계값(99.5\~99.8%) 옵션 정리, 결과 저장 포맷 정리                 |
| 2025-09-03 | v1.0.0 | PatchCore 트립틱/오버레이 연동, 오버레이 튜닝 파라미터(`clip_q`, `gamma` 등) |
| (예정)       | v1.1.0 | 임계값 자동 캘리브레이션(ROC 기반 옵션), 배치 추론/리포트, 모델/데이터 프로필링         |

---

## 📄 라이선스

프로젝트 루트의 `LICENSE` 파일을 확인하세요. (미동봉 시 추후 추가 예정)

---

## 🙌 기여

이슈/PR 환영합니다. 버그 재현용 샘플 이미지와 로그(가능 시 콘솔 출력)를 첨부해 주세요.
