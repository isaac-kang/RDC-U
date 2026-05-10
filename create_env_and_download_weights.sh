#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# rdc-u.py 가 쓰는 모든 외부 의존성을 한 번에 셋업:
#   1) conda env `detectron2` (py3.8 + torch1.9 + detectron2 v0.6)
#   2) DPText-DETR repo + custom adet 빌드 (local-models/DPText-DETR)
#   3) Union14M tools/ sparse clone (rotate_crop import 만 위해 — vestigial)
#   4) DPText-DETR 두 weight (ArT, Total-Text) → Union14M/checkpoints/
#
# 핵심 제약 / 결정:
#   - DPText-DETR (ymy-k repo) 가 detectron2 v0.6 stack 기반 → torch 1.9 + cu111
#     조합 외엔 prebuilt wheel 없음.
#   - AdelaiDet-계열 `python setup.py build develop` 는 setuptools<60 (pkg_resources).
#   - Pillow ≥10 에선 Image.LINEAR 제거 → detectron2 v0.6 transforms 깨짐.
#   - timm 최신 은 torch.fx (torch ≥1.10) 의존 → timm 0.6.x 로 pin.
#   - numba 0.60+ 는 py3.10+ 필요 → numba<0.60.
#   - lmdb wheel 기본은 Py_SET_REFCNT (py3.10+) 씀 → lmdb 1.4.1.
#
# 사전 가정:
#   - GPU 드라이버 ≥ 450 (cu111 호환)
#   - /data/isaackang/anaconda3 에 conda 설치
# -----------------------------------------------------------------------------
set -e

ENV_NAME=detectron2
CONDA_ROOT=/data/isaackang/anaconda3
CONDA_BIN=${CONDA_ROOT}/bin/conda
ENV_PY=${CONDA_ROOT}/envs/${ENV_NAME}/bin/python
ENV_PIP=${CONDA_ROOT}/envs/${ENV_NAME}/bin/pip
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

DPTEXT_DIR=${DPTEXT_DIR:-$SCRIPT_DIR/local-models/DPText-DETR}
DPTEXT_REPO=${DPTEXT_REPO:-https://github.com/ymy-k/DPText-DETR.git}
UNION14M_DIR=${UNION14M_DIR:-$SCRIPT_DIR/Union14M}
UNION14M_REPO=${UNION14M_REPO:-https://github.com/Mountchicken/Union14M.git}
CKPT_DIR="$UNION14M_DIR/checkpoints"
HF_MIRROR_REPO=${HF_MIRROR_REPO:-isaackang/DPText_DETR_R_50_poly}

step() { echo; echo "==[ $* ]=="; }

# ─────────────────────────────────────────────────────────────────────────
# 1) conda env
# ─────────────────────────────────────────────────────────────────────────
step "conda env $ENV_NAME (create if missing)"
if [ ! -x "$ENV_PY" ]; then
  "$CONDA_BIN" create -n "$ENV_NAME" python=3.8 -y
fi

step 'pinned build/runtime deps (setuptools/Pillow/numpy/opencv/lmdb/timm/numba)'
$ENV_PIP install --quiet \
  'setuptools==59.5.0' 'Pillow<10' 'numpy<2' \
  'opencv-python<4.11' 'opencv-python-headless<4.11' \
  'lmdb==1.4.1' 'timm==0.6.13' 'numba<0.60'

step 'PyTorch 1.9.0 + torchvision 0.10.0 (cu111)'
$ENV_PIP install --quiet \
  torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
  --extra-index-url https://download.pytorch.org/whl/cu111

step 'detectron2 v0.6 (prebuilt for torch1.9+cu111)'
$ENV_PIP install --quiet detectron2==0.6 \
  -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html

step 'HuggingFace streaming + rdc-u runtime deps'
$ENV_PIP install --quiet \
  huggingface_hub pyarrow fsspec requests tqdm

# ─────────────────────────────────────────────────────────────────────────
# 2) DPText-DETR repo + adet 빌드
# ─────────────────────────────────────────────────────────────────────────
step "DPText-DETR repo clone ($DPTEXT_DIR)"
mkdir -p "$(dirname "$DPTEXT_DIR")"
if [ ! -d "$DPTEXT_DIR" ]; then
  git clone --depth 1 "$DPTEXT_REPO" "$DPTEXT_DIR"
fi

step 'DPText-DETR adet 빌드 (custom CUDA ops; 첫 빌드 ~수분)'
(
  cd "$DPTEXT_DIR"
  "$ENV_PY" setup.py build develop
) 1>/dev/null

# ─────────────────────────────────────────────────────────────────────────
# 3) Union14M tools/ — rdc-u.py 가 rotate_crop 을 import 함 (현재 미사용이지만
#    import 자체는 살아있음). tools/ 만 sparse clone.
# ─────────────────────────────────────────────────────────────────────────
step "Union14M sparse clone (tools/ only)"
if [ ! -d "$UNION14M_DIR" ]; then
  git clone --depth 1 --filter=blob:none --sparse "$UNION14M_REPO" "$UNION14M_DIR"
  git -C "$UNION14M_DIR" sparse-checkout set tools
  rm -rf "$UNION14M_DIR/.git"
fi
mkdir -p "$CKPT_DIR"

# ─────────────────────────────────────────────────────────────────────────
# 4) DPText-DETR weights (HF mirror, 자동)
# ─────────────────────────────────────────────────────────────────────────
step "DPText-DETR weights 다운로드 (HF mirror: $HF_MIRROR_REPO)"
WEIGHTS=(
  "dptext_art_final.pth|art_final.pth"
  "dptext_totaltext.pth|totaltext.pth"
)
for entry in "${WEIGHTS[@]}"; do
  IFS='|' read -r dest src <<< "$entry"
  dest_path="$CKPT_DIR/$dest"
  if [ -s "$dest_path" ]; then
    echo "  ✓ $dest  (already at $dest_path)"
    continue
  fi
  url="https://huggingface.co/$HF_MIRROR_REPO/resolve/main/$src"
  echo "  ↓ $dest  ← $url"
  wget -q --show-progress -O "$dest_path" "$url"
done

# ─────────────────────────────────────────────────────────────────────────
# verify
# ─────────────────────────────────────────────────────────────────────────
step 'verify imports'
$ENV_PY - <<PY
import sys
import torch, detectron2, numpy, cv2, PIL, timm, lmdb
print(f'python     {sys.version.split()[0]}')
print(f'torch      {torch.__version__}  cuda={torch.cuda.is_available()}')
print(f'detectron2 {detectron2.__version__}')
print(f'numpy      {numpy.__version__}')
print(f'cv2        {cv2.__version__}')
print(f'Pillow     {PIL.__version__}')
print(f'timm       {timm.__version__}')
print(f'lmdb       {lmdb.__version__}')

sys.path.insert(0, '$DPTEXT_DIR')
from adet.config import get_cfg
print('adet       ok (DPText-DETR custom build)')

sys.path.insert(0, '$UNION14M_DIR/tools')
import rotate_crop
print('rotate_crop ok (Union14M tools)')
PY

step 'verify checkpoints'
all_ok=1
for entry in "${WEIGHTS[@]}"; do
  IFS='|' read -r dest _ <<< "$entry"
  dest_path="$CKPT_DIR/$dest"
  if [ -s "$dest_path" ]; then
    sz=$(du -h "$dest_path" | cut -f1)
    echo "  ✓ $dest  ($sz)"
  else
    echo "  ✗ $dest  MISSING"
    all_ok=0
  fi
done

echo
if [ $all_ok -eq 1 ]; then
  echo "[done] $ENV_NAME env + DPText-DETR + weights ready"
  echo
  echo "사용 예:"
  echo "  $ENV_PY rdc-u.py --steps 1,2 --target 500"
  echo "또는:"
  echo "  conda activate $ENV_NAME"
  echo "  python rdc-u.py --steps 1,2 --target 500"
else
  echo "[partial] $ENV_NAME env ready, but some weights missing — "
  echo "네트워크 / HF mirror 접근권한 확인 후 재실행."
  exit 1
fi
