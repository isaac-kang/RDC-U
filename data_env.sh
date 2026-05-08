#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# data env 셋업 기록 (conda env "data")
#
# 용도: HuggingFace 데이터 stream + DBNet++ (mmocr) 텍스트 검출 추론.
#
# 핵심 제약 / 결정:
#   - mmocr 1.0.0 의 mmcv constraint 가 ">=2.0.0rc4,<2.1.0" → mmcv 2.0.x 만 가능
#   - mmcv 2.0.x 의 prebuilt wheel(cp311) 은 cu118/torch2.0 조합에만 존재
#     → torch 2.0.1 + cu118 + mmcv 2.0.1 로 고정
#   - numpy 는 <2 (torch 2.0.1 이 numpy 1.x 빌드)
#   - opencv-python 은 <4.11 (4.11+ 는 numpy>=2 강요)
#   - mmocr setup 이 pkg_resources 쓰는데 setuptools>=70 부터 누락 → setuptools<70
#   - Union14M 가 modify 한 mmocr 가 timm 을 추가로 import
#
# 사전 가정:
#   - conda env "data" 가 이미 존재 (없으면 `conda create -n data python=3.11 -y`)
#   - Union14M repo 가 ~/STR/Union14M 에 clone (mmocr-dev-1.x 포함)
#   - GPU 드라이버 ≥ 525 (cu118 호환)
# -----------------------------------------------------------------------------
set -e

ENV_NAME=data
ENV_PY=/data/isaackang/anaconda3/envs/${ENV_NAME}/bin/python
ENV_PIP=/data/isaackang/anaconda3/envs/${ENV_NAME}/bin/pip
UNION14M_DIR=${UNION14M_DIR:-/data/isaackang/STR/Union14M}

[ -x "$ENV_PY" ] || { echo "env $ENV_NAME not found at $ENV_PY"; exit 1; }

step() { echo; echo "==[ $* ]=="; }

step 'HuggingFace 데이터 stream 도구 (멱등)'
$ENV_PIP install --quiet huggingface_hub pyarrow pillow requests tqdm

step 'numpy<2, opencv<4.11, setuptools<70 (mmocr/torch 빌드 호환)'
$ENV_PIP install --quiet 'numpy<2' 'opencv-python<4.11' 'opencv-python-headless<4.11' 'setuptools<70'

step 'PyTorch 2.0.1 + torchvision 0.15.2 (cu118)'
$ENV_PIP install --quiet torch==2.0.1 torchvision==0.15.2 \
  --index-url https://download.pytorch.org/whl/cu118

step 'mmengine + mmcv 2.0.1 (prebuilt for torch2.0+cu118)'
$ENV_PIP install --quiet 'mmengine>=0.7,<1.0'
$ENV_PIP install --quiet mmcv==2.0.1 \
  -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

step 'mmdet 3.0.x'
$ENV_PIP install --quiet 'mmdet>=3.0.0,<3.1.0'

step 'mmocr (Union14M 안의 local checkout, --no-deps)'
$ENV_PIP install --quiet -e "$UNION14M_DIR/mmocr-dev-1.x" --no-deps

step 'mmocr / Union14M 런타임 deps'
$ENV_PIP install --quiet imgaug rapidfuzz lmdb pyclipper shapely scikit-image timm

step 'verify'
$ENV_PY - <<'PY'
import torch, mmocr, mmcv, mmdet, mmengine, numpy, cv2, timm
print(f'torch    {torch.__version__}  cuda={torch.cuda.is_available()}')
print(f'numpy    {numpy.__version__}')
print(f'cv2      {cv2.__version__}')
print(f'mmengine {mmengine.__version__}')
print(f'mmcv     {mmcv.__version__}')
print(f'mmdet    {mmdet.__version__}')
print(f'mmocr    {mmocr.__version__}')
print(f'timm     {timm.__version__}')
PY

step 'DBNet++ (ResNet50-oCLIP) 가중치'
mkdir -p "$UNION14M_DIR/checkpoints"
WEIGHT="$UNION14M_DIR/checkpoints/dbnetpp_oclip.pth"
if [ ! -s "$WEIGHT" ]; then
  wget -q --show-progress -O "$WEIGHT" \
    https://download.openmmlab.com/mmocr/textdet/dbnetpp/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015_20221101_124139-4ecb39ac.pth
fi
ls -lh "$WEIGHT"

echo
echo "[done] $ENV_NAME env ready. Run:"
echo "  $ENV_PY $UNION14M_DIR/../STR_DATA_Ops/crop_with_dbnetpp.py --n 5"
