"""
DBNet++ (ResNet50-oCLIP, mmocr) 로 텍스트 영역 검출 후 rotate-crop 으로 잘라내기.

입력: 폴더 안 jpg/png 이미지들 (디폴트 ./recap_str_samples)
출력:
  - <out>/crops/<imgname>__<idx>.jpg : crop 이미지들
  - <out>/<out_name>_preview.html    : 원본(폴리곤 오버레이) + crops 시각화

준비: data_env.sh 한번 돌려 mmocr 스택 설치 완료 상태여야 함.

사용법:
    python crop_with_dbnetpp.py                       # 디폴트 5장
    python crop_with_dbnetpp.py --n 10 --score_thr 0.3
    python crop_with_dbnetpp.py --in_dir ./other_dir --out ./other_crops
"""

import argparse
import base64
import html
import io
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


UNION14M_DIR = Path('/data/isaackang/STR/Union14M')
DEFAULT_CFG = UNION14M_DIR / 'mmocr-dev-1.x/configs/textdet/dbnetpp/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015.py'
DEFAULT_WEIGHTS = UNION14M_DIR / 'checkpoints/dbnetpp_oclip.pth'

# Union14M 의 rotate_crop 재사용 (직접 import 하기 위해 path 추가)
sys.path.insert(0, str(UNION14M_DIR / 'tools'))
import rotate_crop as _rc  # noqa: E402

# rotate_crop.find_long_side_points 에 wrap-around 버그 있음:
# longest side index 가 마지막일 때 polygon[idx+1] 이 OOB. 패치.
_orig_find_long_side = _rc.find_long_side_points
def _patched_find_long_side(polygon):
    p = polygon.reshape(-1, 2).astype(np.float32)
    p_close = np.concatenate((p, p[:1]), axis=0)
    edges = np.linalg.norm(p_close[1:] - p_close[:-1], axis=1)
    idx = int(np.argmax(edges))
    n = len(polygon)
    return polygon[idx], polygon[(idx + 1) % n]
_rc.find_long_side_points = _patched_find_long_side
rotate_crop = _rc.rotate_crop


def axis_aligned_crop(img: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """polygon 의 bounding box 로 axis-aligned crop (rotate_crop 실패 fallback)."""
    pts = polygon.reshape(-1, 2).astype(np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    H, W = img.shape[:2]
    x = max(0, x); y = max(0, y)
    return img[y:min(H, y + h), x:min(W, x + w)]


def expand(p: str) -> Path:
    return Path(p).expanduser().resolve()


def overlay_polygons(img_bgr: np.ndarray, polys: list[np.ndarray]) -> np.ndarray:
    """원본 위에 polygon 그려서 시각화용 BGR 반환."""
    vis = img_bgr.copy()
    for poly in polys:
        pts = poly.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 220, 0), thickness=2)
    return vis


def encode_jpeg(img_bgr: np.ndarray, quality: int = 88) -> str:
    ok, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ''
    return base64.b64encode(buf.tobytes()).decode()


def render_html(items, out_dir: Path) -> str:
    """items: list[dict(name, vis_b64, crops=[(score, b64)])]."""
    rows = []
    for it in items:
        crops_html = ''.join(
            f'<div class="crop">'
            f'  <img src="data:image/jpeg;base64,{c["b64"]}"/>'
            f'  <div class="sc">{c["score"]:.2f}</div>'
            f'</div>'
            for c in it['crops']
        ) or '<div class="empty">(no detections)</div>'
        rows.append(
            f'<div class="row">'
            f'  <div class="img"><img src="data:image/jpeg;base64,{it["vis_b64"]}"/>'
            f'    <div class="cap">{html.escape(it["name"])} · {len(it["crops"])} det</div>'
            f'  </div>'
            f'  <div class="crops">{crops_html}</div>'
            f'</div>'
        )
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"/>
<title>DBNet++ crops</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 32px auto; padding: 0 24px;
          background: #fff; color: #222; max-width: 1400px; }}
  h1 {{ font-size: 18px; margin: 0 0 24px; }}
  .row {{ display: grid; grid-template-columns: 480px 1fr; gap: 24px; padding: 24px 0; }}
  .row + .row {{ border-top: 1px solid #eee; }}
  .img img {{ width: 480px; max-height: 480px; object-fit: contain; background: #fafafa; display: block; }}
  .cap {{ font-size: 11px; color: #888; margin-top: 4px; font-family: ui-monospace, Menlo, monospace; }}
  .crops {{ display: flex; flex-wrap: wrap; gap: 8px; align-content: flex-start; }}
  .crop {{ background: #fafafa; padding: 4px; border: 1px solid #eee; }}
  .crop img {{ max-height: 60px; max-width: 320px; display: block; }}
  .crop .sc {{ font-size: 10px; color: #999; text-align: right; font-family: ui-monospace, Menlo, monospace; }}
  .empty {{ color: #ccc; font-size: 13px; padding: 12px; }}
</style></head>
<body>
<h1>DBNet++ (ResNet50-oCLIP) crops · {len(rows)} images, saved to <code>{html.escape(str(out_dir))}</code></h1>
{''.join(rows)}
</body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in_dir', type=expand, default=expand('./recap_str_samples'),
                    help='입력 이미지 폴더')
    ap.add_argument('--out', type=expand, default=expand('./recap_str_crops'),
                    help='출력 폴더 (crops/ + preview.html 가 여기에 생성)')
    ap.add_argument('--n', type=int, default=5, help='처리할 이미지 수 (default 5)')
    ap.add_argument('--config', type=expand, default=DEFAULT_CFG)
    ap.add_argument('--weights', type=expand, default=DEFAULT_WEIGHTS)
    ap.add_argument('--score_thr', type=float, default=0.3,
                    help='polygon score threshold')
    ap.add_argument('--device', type=str, default='cuda:0')
    args = ap.parse_args()

    if not args.config.exists():
        sys.exit(f'[error] config not found: {args.config}')
    if not args.weights.exists():
        sys.exit(f'[error] weights not found: {args.weights}')
    if not args.in_dir.exists():
        sys.exit(f'[error] in_dir not found: {args.in_dir}')

    img_paths = sorted(p for p in args.in_dir.iterdir()
                       if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})
    img_paths = img_paths[:args.n]
    if not img_paths:
        sys.exit(f'[error] no images in {args.in_dir}')

    args.out.mkdir(parents=True, exist_ok=True)
    crops_dir = args.out / 'crops'
    crops_dir.mkdir(exist_ok=True)

    print(f'config  : {args.config.name}')
    print(f'weights : {args.weights.name}')
    print(f'images  : {len(img_paths)} from {args.in_dir}')
    print(f'out     : {args.out}')
    print(f'thr     : {args.score_thr}')

    # detector init (mmocr inference 는 import 가 무겁고 stderr 노이즈 있음)
    from mmocr.apis import TextDetInferencer
    det = TextDetInferencer(model=str(args.config),
                            weights=str(args.weights),
                            device=args.device)

    items = []
    total_crops = 0
    for ip in tqdm(img_paths, desc='detect+crop'):
        img_bgr = cv2.imread(str(ip))
        if img_bgr is None:
            print(f'[skip] cannot read {ip}')
            continue

        result = det(str(ip), return_vis=False)
        pred = result['predictions'][0]
        polys_raw = pred.get('polygons', [])  # list[list[float]] flat (x1,y1,x2,y2,...)
        scores = pred.get('scores', [])

        crops = []
        polys_arr = []
        for poly_flat, sc in zip(polys_raw, scores):
            if sc < args.score_thr:
                continue
            poly = np.asarray(poly_flat, dtype=np.float32).reshape(-1, 2)
            polys_arr.append(poly)
            try:
                crop = rotate_crop(img_bgr.copy(), poly.copy())
            except Exception:
                crop = axis_aligned_crop(img_bgr, poly)
            if crop is None or crop.size == 0 or min(crop.shape[:2]) < 4:
                continue
            ci = len(crops)
            cp = crops_dir / f'{ip.stem}__{ci:02d}.jpg'
            cv2.imwrite(str(cp), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            crops.append({'score': float(sc), 'b64': encode_jpeg(crop)})

        vis = overlay_polygons(img_bgr, polys_arr)
        items.append({'name': ip.name,
                      'vis_b64': encode_jpeg(vis),
                      'crops': crops})
        total_crops += len(crops)

    html_path = args.out.parent / f'{args.out.name}_preview.html'
    html_path.write_text(render_html(items, args.out), encoding='utf-8')
    print(f'\n{total_crops} crops over {len(items)} images')
    print(f'crops/  : {crops_dir}')
    print(f'preview : {html_path}')


if __name__ == '__main__':
    main()
