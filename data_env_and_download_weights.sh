#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# rdc-u.py 가 쓰는 외부 의존성 가져오기:
#   - Union14M/tools/rotate_crop.py  (sparse clone)
#   - DPText-DETR ArT finetune   weight   →  Union14M/checkpoints/dptext_art_final.pth
#   - DPText-DETR Total-Text     weight   →  Union14M/checkpoints/dptext_totaltext.pth
#
# 두 weight 다 isaackang HF mirror 에서 wget 자동. mmocr/mmcv 같은 무거운 env 설정은
# 더 이상 안 함 — rdc-u.py 는 detectron2 env (별도) 에서 돌고, mmocr 의존성 없음.
# -----------------------------------------------------------------------------
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
UNION14M_DIR=${UNION14M_DIR:-$SCRIPT_DIR/Union14M}
UNION14M_REPO=${UNION14M_REPO:-https://github.com/Mountchicken/Union14M.git}
CKPT_DIR="$UNION14M_DIR/checkpoints"
HF_MIRROR_REPO=${HF_MIRROR_REPO:-isaackang/DPText_DETR_R_50_poly}

step() { echo; echo "==[ $* ]=="; }

step "Union14M sparse clone (rotate_crop tool only)"
if [ ! -d "$UNION14M_DIR" ]; then
  git clone --depth 1 --filter=blob:none --sparse "$UNION14M_REPO" "$UNION14M_DIR"
  git -C "$UNION14M_DIR" sparse-checkout set tools
  rm -rf "$UNION14M_DIR/.git"
fi
mkdir -p "$CKPT_DIR"

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

step "verify"
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
  echo "[done] DPText-DETR weights ready at $CKPT_DIR"
else
  echo "[partial] 위 MISSING 항목 다시 확인 (네트워크 / mirror 접근권한)."
  exit 1
fi
