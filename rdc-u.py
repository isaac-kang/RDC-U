"""
RDC-U: Recap-DataComp → STR-U (unlabeled) crop 데이터셋 통합 framework.

stages — `--steps` 에 comma-separated 로 조합:
  filter : 모든 train shard 의 caption-pass row index 를 .npy 로 저장하고,
           일부 filtered row 는 URL 다운로드 → filter_preview.html 로 시각 검수
  crop   : (filter index 사용) URL → memory → DPText-DETR (ArT + Total-Text,
           둘 다 R50) raw output 합산 → AABB-NMS (IoU>0.5, top-100) →
           AABB crop → LMDB.
           Union14M-U 호환 unlabeled LMDB (image-NNNNNNNNN + num-samples,
           label key 없음 — SLD 의 unlabeled auto-detect 와 호환). 개별 jpg 는
           --save_crops 로 추가 dump. 시각 검수는 --crop_preview_n.

step 2(crop) 는 step 1 결과 (.npy index) 를 사용함 — index 없는 shard 는 skip.
default flow (--steps 미지정) 는 1 → 2 순으로 돌아 filter + crop 을 수행함.

env: detectron2 (py3.8 + torch1.9 + detectron2 v0.6 + AdelaiDet/DPText-DETR setup).
mmocr 의존성 제거됨.

step alias: 1=filter, 2=crop. `index` 는 filter 의 legacy 이름으로 허용.
--steps 생략 시 filter,crop (1,2) 실행.

기본 동작은 small crop 모드 (max_shards=1, target=100). 풀 실행은
  `--full` 또는 `--max_shards 0 --target 0` 명시.

usage:
    python rdc-u.py                                      # 1+2 small crop
    python rdc-u.py --steps 1                            # filter index + preview
    python rdc-u.py --steps 2                            # 1 shard, 100 crops
    python rdc-u.py --steps 1,2                          # filter + crop
    python rdc-u.py --full                               # 전체 데이터 filter + crop
    python rdc-u.py --steps 2 --max_shards 0 --target 100000   # cap 만

multi-GPU:
    --num_gpus 0  → torch.cuda.device_count() 만큼 STD inference 병렬.
multi-machine:
    --shard_mod M --shard_rem R  로 머신 단위 분할.
"""

import time
SCRIPT_T0 = time.time()

import resource as _resource
import sys as _sys_for_fd


def _bump_fd_limit():
    """RLIMIT_NOFILE soft → hard (또는 65536) 까지 자동 상향.

    282 shards × parquet/.npy reader + 128 fetch threads + lmdb 등이 동시에
    파일을 잡으면 default soft 1024 로 OSError [Errno 24] Too many open files
    발생. hard limit 까지는 root 없이도 set 가능. 실패해도 fatal 아니라 warn 만.
    """
    try:
        soft, hard = _resource.getrlimit(_resource.RLIMIT_NOFILE)
        target = hard if hard > 0 else 65536
        if soft < target:
            _resource.setrlimit(_resource.RLIMIT_NOFILE, (target, hard))
    except Exception as e:
        print(f'[warn] could not raise RLIMIT_NOFILE: {e}',
              file=_sys_for_fd.stderr)


_bump_fd_limit()


import argparse
import base64
import html
import io
import json
import random
import re
import shutil
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from queue import Queue

import cv2
import lmdb
import numpy as np
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from huggingface_hub import HfFileSystem
from PIL import Image
from tqdm import tqdm

# 풀 실행 시 fetch (128 threads) · post worker · GPU 가 동시에 cv2 ops 호출. cv2 가
# 자체 thread 를 spawn 하면 oversubscription. 우리 thread pool 안에서만 돌게 0 으로 set.
cv2.setNumThreads(0)

# Union14M tools (rotate_crop) + wrap-around 버그 패치
UNION14M_DIR = Path(__file__).resolve().parent / 'Union14M'
sys.path.insert(0, str(UNION14M_DIR / 'tools'))
import rotate_crop as _rc  # noqa: E402


def _patched_find_long_side(polygon):
    p = polygon.reshape(-1, 2).astype(np.float32)
    p_close = np.concatenate((p, p[:1]), axis=0)
    edges = np.linalg.norm(p_close[1:] - p_close[:-1], axis=1)
    idx = int(np.argmax(edges))
    n = len(polygon)
    return polygon[idx], polygon[(idx + 1) % n]
_rc.find_long_side_points = _patched_find_long_side
rotate_crop = _rc.rotate_crop


# ─────────────────────────────────────────────────────────────────────────
# 공통 상수 / 정규식
# ─────────────────────────────────────────────────────────────────────────

DEFAULT_DS = 'UCSC-VLAA/Recap-DataComp-1B'

# DPText-DETR (AAAI'23) 두 finetune (ArT + Total-Text, 둘 다 R50). DETR 계열이라
# 자체 NMS 가 없고 num_queries 별 score thresholding 만. 두 model 의 raw output 을
# 합쳐서 AABB-NMS (IoU>0.5) + top-100 cap → ArT 의 회전·곡선·inverse + TT 의 curved
# scene 을 모두 cover.
RDC_ROOT = Path(__file__).resolve().parent
DPTEXT_REPO = RDC_ROOT / 'local-models' / 'DPText-DETR'
DPTEXT_MODELS = [
    ('dptext_art',
     'configs/DPText_DETR/ArT/R_50_poly.yaml',
     'Union14M/checkpoints/dptext_art_final.pth'),
    ('dptext_tt',
     'configs/DPText_DETR/TotalText/R_50_poly.yaml',
     'Union14M/checkpoints/dptext_totaltext.pth'),
]
UA = 'Mozilla/5.0 (compatible; RDC-U/1.0)'
END = object()  # consumer 종료 sentinel

# 인용부호 안 string — 이미지 속 실제 텍스트일 가능성 높음 (LLaVA 패턴).
QUOTED_RE = re.compile(
    r'[\"“]([^\"“”\n]+?)[\"”]'
    r"|"
    r"(?<!\w)['‘]([^'‘’\n]+?)['’](?!\w)"
)

# tier-1 신호 — quoted 없어도 강한 signal
TIER1_RE = re.compile(
    r"\b("
    r"text|word|words|"
    r"reads|says|displays|"
    r"written|titled|labeled|labelled|spelling|"
    r"inscription|inscribed|engraved"
    r")\b",
    re.IGNORECASE,
)


def caption_has_str_signal(cap: str) -> bool:
    if not cap:
        return False
    return bool(TIER1_RE.search(cap) or QUOTED_RE.search(cap))


def expand(p: str) -> Path:
    return Path(p).expanduser().resolve()


def shard_idx_of(name: str) -> int:
    """train-00042-of-04627 → 42"""
    try:
        return int(name.split('-')[1])
    except Exception:
        return -1


def _load_shard_index(out_root: Path, shard_name: str):
    """step 1 의 산출물 <out_root>/index/<shard>.npy 로딩. 없으면 None."""
    p = out_root / 'index' / f'{shard_name}.npy'
    if not p.exists():
        return None
    return np.load(p)


def _filter_paths_with_index(paths, out_root: Path):
    """index .npy 가 존재하는 shard 만 남김. (kept_paths, missing_count) 반환."""
    kept = []
    missing = 0
    for p in paths:
        if (out_root / 'index' / f'{Path(p).stem}.npy').exists():
            kept.append(p)
        else:
            missing += 1
    return kept, missing


_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp')


def _decode_bytes_to_bgr(data: bytes):
    """JPEG/PNG/WebP bytes → cv2 BGR np.ndarray. 실패 시 None.

    cv2.imdecode (libjpeg-turbo) 가 PIL.open+convert+np.array 보다 2~3x 빠르고
    중간 copy 도 없음. cv2 가 못 읽는 포맷 (animated GIF 등) 만 PIL fallback.
    """
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        pil = Image.open(io.BytesIO(data)).convert('RGB')
        return np.array(pil)[:, :, ::-1].copy()
    except Exception:
        return None


def _is_imageish(content_type: str, url: str) -> bool:
    if content_type and 'image' in content_type:
        return True
    return url.lower().split('?')[0].endswith(_IMG_EXTS)


# Per-thread requests.Session — TCP/TLS connection 재사용 + DNS 캐시 효과로
# fetch 128 threads 환경에서 매번 new connect 하던 비용 제거.
_session_local = threading.local()


def _get_session():
    s = getattr(_session_local, 'session', None)
    if s is not None:
        return s
    s = requests.Session()
    s.headers['User-Agent'] = UA
    adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=0)
    s.mount('http://', adapter)
    s.mount('https://', adapter)
    _session_local.session = s
    return s


def fetch_image_bgr(url: str, timeout):
    """URL → cv2 BGR np.ndarray. 실패 시 None.

    timeout: float 또는 (connect, read) tuple.
    """
    try:
        r = _get_session().get(url, timeout=timeout)
        r.raise_for_status()
        if not _is_imageish(r.headers.get('Content-Type', ''), url):
            return None
        return _decode_bytes_to_bgr(r.content)
    except Exception:
        return None


def axis_aligned_crop(img: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    pts = polygon.reshape(-1, 2).astype(np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    H, W = img.shape[:2]
    x = max(0, x); y = max(0, y)
    return img[y:min(H, y + h), x:min(W, x + w)]


def _poly_aabb(pts):
    """polygon (N,2) → (x1,y1,x2,y2) AABB. <3 점이면 None."""
    arr = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if len(arr) < 3:
        return None
    x1 = float(arr[:, 0].min())
    y1 = float(arr[:, 1].min())
    x2 = float(arr[:, 0].max())
    y2 = float(arr[:, 1].max())
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _aabb_iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def aabb_nms_combined(entries, iou_thr: float, max_keep: int, rng=None):
    """Greedy AABB-NMS — IoU 측정만 AABB. NMS 자체는 score 내림차순 (overlap group
    에서 best 를 keep) 이지만, NMS 통과 detection 이 max_keep 보다 많으면 score top-K
    가 아니라 *랜덤* K 개 sampling.

    entries: list of {'poly': np.ndarray (N,2), 'score': float}.
    rng: random.Random — None 이면 module-level random 사용.
    return: list of (x1, y1, x2, y2, score) — keep 된 detection 의 AABB.
    """
    if not entries:
        return []
    items = []
    for e in entries:
        bb = _poly_aabb(e['poly'])
        if bb is None:
            continue
        items.append((float(e['score']), bb))
    # NMS step: score 내림차순으로 dedupe (overlap 시 high-score 가 winner)
    items.sort(key=lambda t: -t[0])
    keep = []
    for sc, bb in items:
        ok = True
        for ksc, kbb in keep:
            if _aabb_iou(bb, kbb) > iou_thr:
                ok = False
                break
        if ok:
            keep.append((sc, bb))
    # max_keep cap: 랜덤 sampling — score 분포에 편향되지 않도록.
    if max_keep > 0 and len(keep) > max_keep:
        sampler = rng if rng is not None else random
        keep = sampler.sample(keep, max_keep)
    return [(bb[0], bb[1], bb[2], bb[3], sc) for sc, bb in keep]


def overlay_polygons(img_bgr: np.ndarray, polys, scores=None) -> np.ndarray:
    vis = img_bgr.copy()
    H, W = vis.shape[:2]
    for i, poly in enumerate(polys):
        pts = poly.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 220, 0), thickness=2)
        if scores is None:
            continue
        xy = poly.reshape(-1, 2)
        x = int(max(0, xy[:, 0].min()))
        y = int(xy[:, 1].min())
        text = f'{scores[i]:.2f}'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = y - 4 if y - th - 4 >= 0 else min(H - 2, y + th + 4)
        bg_y0 = max(0, ty - th - 2)
        bg_y1 = min(H, ty + 2)
        bg_x1 = min(W, x + tw + 4)
        cv2.rectangle(vis, (x, bg_y0), (bg_x1, bg_y1), (0, 0, 0), -1)
        cv2.putText(vis, text, (x + 2, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1, cv2.LINE_AA)
    return vis


# DPText-DETR per-model overlay 색 (BGR)
_DETECTOR_COLORS = [
    (3, 255, 118),    # ArT — bright green (#76ff03)
    (23, 221, 100),   # Total-Text — darker green (#64dd17)
]


def overlay_ensemble(img_bgr, polys_per_det, consensus_boxes):
    """Per-detector raw polygon (얇은 색별 선) + final AABB (굵은 흰색)."""
    vis = img_bgr.copy()
    for color, polys in zip(_DETECTOR_COLORS, polys_per_det):
        for poly in polys:
            pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=1)
    for (x1, y1, x2, y2, sc) in consensus_boxes:
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(vis, p1, p2, (255, 255, 255), 2)
        text = f'{sc:.2f}'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        x, y = p1
        ty = y - 4 if y - th - 4 >= 0 else min(vis.shape[0] - 2, y + th + 4)
        cv2.rectangle(vis, (x, max(0, ty - th - 2)),
                      (min(vis.shape[1], x + tw + 4), min(vis.shape[0], ty + 2)),
                      (0, 0, 0), -1)
        cv2.putText(vis, text, (x + 2, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def encode_jpeg(img_bgr: np.ndarray, quality: int = 88) -> str:
    ok, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode() if ok else ''


def fmt_secs(s: float) -> str:
    if s < 60:
        return f'{s:.1f}s'
    if s < 3600:
        return f'{s/60:.1f}min ({s:.0f}s)'
    return f'{s/3600:.2f}h ({s:.0f}s)'


# ═════════════════════════════════════════════════════════════════════════
# step: filter — 모든 shard caption-pass row index .npy + preview
# ═════════════════════════════════════════════════════════════════════════

def _index_shard(fs, shard_path, out_dir, force):
    name = Path(shard_path).stem
    out_file = out_dir / f'{name}.npy'
    if out_file.exists() and not force:
        arr = np.load(out_file)
        return {'shard': name, 'rows': None, 'matches': int(arr.size), 'cached': True}
    matches = []
    offset = 0
    with fs.open(shard_path, 'rb') as f:
        pf = pq.ParquetFile(f)
        for rg_idx in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg_idx, columns=['re_caption'])
            caps = tbl.column('re_caption').to_pylist()
            for i, c in enumerate(caps):
                if caption_has_str_signal(c or ''):
                    matches.append(offset + i)
            offset += len(caps)
    arr = np.asarray(matches, dtype=np.uint32)
    np.save(out_file, arr)
    return {'shard': name, 'rows': offset, 'matches': int(arr.size), 'cached': False}


def run_index(args):
    stage_t0 = time.time()
    out_dir = args.out_root / 'index'
    out_dir.mkdir(parents=True, exist_ok=True)

    fs = HfFileSystem()
    paths = sorted(fs.glob(f'datasets/{args.dataset}/data/train_data/*.parquet'))
    if args.max_shards:
        paths = paths[:args.max_shards]

    print(f'[filter] dataset : {args.dataset}')
    print(f'[filter] shards  : {len(paths)} (workers {args.workers})')
    print(f'[filter] index   : {out_dir}')

    t0 = time.time()
    matched_total = 0
    rows_total = 0
    n_done = 0
    n_cached = 0
    pbar = tqdm(total=len(paths), desc='filter', smoothing=0.05)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_index_shard, fs, p, out_dir, args.force) for p in paths]
        for fut in as_completed(futures):
            r = fut.result()
            matched_total += r['matches']
            if r.get('cached'):
                n_cached += 1
            if r['rows'] is not None:
                rows_total += r['rows']
            n_done += 1
            pbar.set_postfix({'matched': f'{matched_total/1e6:.1f}M'})
            pbar.update(1)
    pbar.close()

    wall = time.time() - t0
    rows_scanned = rows_total if rows_total else None
    match_rate = (matched_total / rows_total) if rows_total else None
    summary = {
        'dataset': args.dataset,
        'shards_processed': n_done,
        'shards_total': len(paths),
        'shards_cached': n_cached,
        'rows_scanned': rows_scanned,
        'matched_total': matched_total,
        'match_rate': match_rate,
        'wall_seconds': round(wall, 1),
        'workers': args.workers,
    }
    print(f'[filter] timing  : wall {fmt_secs(wall)}')
    rows_txt = f'{rows_total:,}' if rows_total else 'unknown (cached index)'
    rate_txt = f' ({100 * match_rate:.2f}%)' if match_rate is not None else ''
    cache_txt = f', cached shards {n_cached}/{n_done}' if n_cached else ''
    print(f'[filter] rows    : {rows_txt}{cache_txt}')
    print(f'[filter] matched : {matched_total:,}{rate_txt}')
    preview_summary = write_filter_preview(args, fs=fs, paths=paths)
    return {
        'step': 'filter',
        'wall': time.time() - stage_t0,
        'scan_wall': wall,
        'rows': rows_scanned,
        'matched': matched_total,
        'match_rate': match_rate,
        'cached': n_cached,
        'shards': n_done,
        'preview': preview_summary,
    }


# ═════════════════════════════════════════════════════════════════════════
# filter preview — N 장 URL 다운로드 + caption highlight HTML
# ═════════════════════════════════════════════════════════════════════════

def _escape_with_highlight(text: str) -> str:
    spans = []
    for m in QUOTED_RE.finditer(text):
        spans.append((m.start(), m.end(), 'q'))
    for m in TIER1_RE.finditer(text):
        spans.append((m.start(), m.end(), 'v'))
    if not spans:
        return html.escape(text)
    spans.sort()
    merged = []
    last_end = -1
    for s, e, c in spans:
        if s < last_end:
            continue
        merged.append((s, e, c))
        last_end = e
    out, last = [], 0
    for s, e, c in merged:
        out.append(html.escape(text[last:s]))
        out.append(f'<span class="hl-{c}">{html.escape(text[s:e])}</span>')
        last = e
    out.append(html.escape(text[last:]))
    return ''.join(out)


def _render_filter_preview_html(items, ds_name, out_dir) -> str:
    rows = []
    for i, it in enumerate(items):
        if it.get('fp') is not None:
            b64 = base64.b64encode(it['fp'].read_bytes()).decode()
        else:
            b64 = encode_jpeg(it['img'], quality=92)
        rows.append(
            f'<div class="row">'
            f'  <div class="idx">{i:03d}</div>'
            f'  <div class="img"><img src="data:image/jpeg;base64,{b64}" /></div>'
            f'  <div class="text">'
            f'    <div class="lbl">re_caption</div>'
            f'    <div class="re">{_escape_with_highlight(it["re_caption"])}</div>'
            f'    <div class="lbl">org_caption</div>'
            f'    <div class="org">{html.escape(it["org_caption"])}</div>'
            f'    <div class="src"><a href="{html.escape(it["url"])}" target="_blank" rel="noreferrer">{html.escape(it["name"])}</a></div>'
            f'  </div>'
            f'</div>'
        )
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8" />
	<title>RDC-U filter preview · {html.escape(ds_name)}</title>
<style>
  :root {{ --col-idx: 40px; --col-img: 320px; --gap: 24px; }}
  body {{ font-family: -apple-system, sans-serif; margin: 32px auto; padding: 0 24px;
          background: #fff; color: #222; max-width: 1280px; }}
  h1 {{ font-size: 18px; margin: 0 0 4px; font-weight: 600; }}
  .topmeta {{ color: #888; font-size: 13px; margin-bottom: 28px; }}
  .row {{ display: grid; grid-template-columns: var(--col-idx) var(--col-img) 1fr;
          column-gap: var(--gap); padding: 28px 0; }}
  .row + .row {{ border-top: 1px solid #eee; }}
  .idx {{ color: #c8c8c8; font-family: ui-monospace, Menlo, monospace;
          font-size: 12px; padding-top: 4px; }}
  .img img {{ width: var(--col-img); max-height: 320px; object-fit: contain;
              background: #fafafa; display: block; }}
  .text {{ font-size: 14px; line-height: 1.6; min-width: 0; }}
  .lbl {{ font-size: 10px; color: #aaa; text-transform: uppercase; letter-spacing: .08em;
          margin: 0 0 6px; font-weight: 600; }}
  .re {{ color: #111; margin-bottom: 18px; white-space: pre-wrap; }}
  .hl-q {{ background: #fff3a3; padding: 0 2px; border-radius: 2px; }}
  .hl-v {{ color: #b85c00; font-weight: 600; }}
  .org {{ color: #777; font-size: 13px; margin-bottom: 18px; white-space: pre-wrap; }}
  .src {{ font-size: 11px; color: #bbb; font-family: ui-monospace, Menlo, monospace; }}
  .src a {{ color: #bbb; text-decoration: none; }}
  .src a:hover {{ color: #555; text-decoration: underline; }}
</style></head>
<body>
	<h1>RDC-U filter preview · <code>{html.escape(ds_name)}</code></h1>
	<div class="topmeta">{len(rows)} filtered images{(' · saved to <code>' + html.escape(str(out_dir)) + '</code>') if out_dir else ''}</div>
{"".join(rows)}
</body></html>
"""


def write_filter_preview(args, fs=None, paths=None):
    n = args.filter_preview_n
    if n <= 0:
        return {
            'wall': 0.0, 'collect_wall': 0.0, 'fetch_wall': 0.0,
            'fetch_sum': 0.0, 'saved': 0,
            'target': 0, 'attempts': 0, 'html': None,
        }

    save_imgs = args.save_preview_images
    out_dir = args.out_root / 'filter_preview'
    if save_imgs:
        out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[filter-preview] dataset : {args.dataset}')
    print(f'[filter-preview] n       : {n}')
    print(f'[filter-preview] threads : {args.filter_preview_threads}')
    if save_imgs:
        print(f'[filter-preview] out_dir : {out_dir} (rows = step1 filter index)')
    else:
        print('[filter-preview] out_dir : (html only; --save_preview_images to dump jpgs)')

    rng = random.Random(args.seed)
    if fs is None:
        fs = HfFileSystem()
    if paths is None:
        paths = sorted(fs.glob(f'datasets/{args.dataset}/data/train_data/*.parquet'))
        if args.max_shards:
            paths = paths[:args.max_shards]
    paths, missing = _filter_paths_with_index(paths, args.out_root)
    if not paths:
        print(f'[filter-preview] no index at {args.out_root}/index/. '
              f'run --steps 1 first (or just `python rdc-u.py`).')
        return {
            'wall': 0.0, 'collect_wall': 0.0, 'fetch_wall': 0.0,
            'fetch_sum': 0.0, 'saved': 0,
            'target': n, 'attempts': 0, 'html': None,
        }
    if missing:
        print(f'[filter-preview] using {len(paths)} indexed shards '
              f'(skipping {missing})')
    rng.shuffle(paths)

    candidates = []
    t0 = time.time()

    for shard_path in paths:
        if len(candidates) >= args.max_attempts:
            break
        shard_name = Path(shard_path).stem
        matched = _load_shard_index(args.out_root, shard_name)
        if matched is None or matched.size == 0:
            continue
        try:
            f = fs.open(shard_path, 'rb')
            pf = pq.ParquetFile(f)
        except Exception as e:
            print(f'[skip {shard_name}] open fail: {e}')
            continue
        try:
            # N 작아서 row group 단위 sparse 접근 (matched 만)
            offset_rg = 0
            rg_jobs = []  # (rg_idx, rg_size, local_positions)
            for rg in range(pf.num_row_groups):
                rg_size = pf.metadata.row_group(rg).num_rows
                lo = int(np.searchsorted(matched, offset_rg))
                hi = int(np.searchsorted(matched, offset_rg + rg_size))
                if lo < hi:
                    positions = (matched[lo:hi] - offset_rg).tolist()
                    rg_jobs.append((rg, offset_rg, positions))
                offset_rg += rg_size
            rng.shuffle(rg_jobs)
            for rg, rg_offset, positions in rg_jobs:
                if len(candidates) >= args.max_attempts:
                    break
                tbl = pf.read_row_group(
                    rg, columns=['url', 're_caption', 'org_caption'])
                urls = tbl.column('url').to_pylist()
                caps = tbl.column('re_caption').to_pylist()
                orgs = tbl.column('org_caption').to_pylist()
                rng.shuffle(positions)
                for pos in positions:
                    if len(candidates) >= args.max_attempts:
                        break
                    url = urls[pos]
                    if not url:
                        continue
                    candidates.append({
                        'url': url,
                        're_caption': caps[pos] or '',
                        'org_caption': orgs[pos] or '',
                    })
        finally:
            f.close()

    items = []
    pbar = tqdm(total=n, desc='filter-preview')
    t_fetch = 0.0
    attempts = 0
    collect_wall = time.time() - t0
    fetch_t0 = time.time()
    workers = max(1, min(args.filter_preview_threads, n, max(len(candidates), 1)))

    def _fetch_candidate(cand):
        t1 = time.time()
        img_bgr = fetch_image_bgr(cand['url'], args.timeout)
        return cand, img_bgr, time.time() - t1

    def _write_item(cand, img_bgr):
        idx = len(items)
        name = f'recap_{idx:05d}.jpg'
        fp = None
        if save_imgs:
            fp = out_dir / name
            cv2.imwrite(str(fp), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        items.append({
            'name': name,
            'fp': fp,
            'img': img_bgr,
            'url': cand['url'],
            're_caption': cand['re_caption'],
            'org_caption': cand['org_caption'],
        })

    with ThreadPoolExecutor(max_workers=workers) as ex:
        candidate_iter = iter(candidates)
        pending = set()
        stop_submitting = False

        def _submit_one():
            nonlocal attempts
            cand = next(candidate_iter)
            pending.add(ex.submit(_fetch_candidate, cand))
            attempts += 1

        for _ in range(workers):
            try:
                _submit_one()
            except StopIteration:
                break

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                cand, img_bgr, dt = fut.result()
                t_fetch += dt
                if len(items) < n and img_bgr is not None:
                    _write_item(cand, img_bgr)
                    pbar.update(1)
                    if len(items) >= n:
                        stop_submitting = True

            while not stop_submitting and len(pending) < workers:
                try:
                    _submit_one()
                except StopIteration:
                    break
    pbar.close()

    fetch_wall = time.time() - fetch_t0
    wall = time.time() - t0
    print(f'[filter-preview] saved   : {len(items)}/{n} '
          f'(after {attempts} attempts)')
    print(f'[filter-preview] timing  : wall {fmt_secs(wall)} · '
          f'collect {fmt_secs(collect_wall)} · '
          f'fetch wall {fmt_secs(fetch_wall)} · '
          f'fetch sum {fmt_secs(t_fetch)} · {workers} threads')

    html_path = None
    if items:
        html_path = args.out_root / 'filter_preview.html'
        html_path.write_text(
            _render_filter_preview_html(items, args.dataset,
                                        out_dir if save_imgs else None),
            encoding='utf-8')
        print(f'[filter-preview] html    : {html_path}')
    return {
        'wall': wall,
        'collect_wall': collect_wall,
        'fetch_wall': fetch_wall,
        'fetch_sum': t_fetch,
        'saved': len(items),
        'target': n,
        'attempts': attempts,
        'threads': workers,
        'html': str(html_path) if html_path else None,
    }


# ═════════════════════════════════════════════════════════════════════════
# step: crop — URL → memory → DPText-DETR (ArT + TT) raw → AABB-NMS → AABB crop + LMDB
# ═════════════════════════════════════════════════════════════════════════

def _iter_shard_metas(shard_paths, fs, out_root, stop_event, stats=None,
                      progress_rows=None):
    """yields (shard_name, metas_chunk). meta = {'shard','row','url'}.

    Q3.1: row group 단위 yield — 한 shard 의 첫 RG 만 읽고도 producer 가
    바로 fetch 시작 가능. caller 는 같은 shard_name 이 연속해서 여러 번
    yield 될 수 있음을 가정해야 함 (transition = shard_name 변경 시점).

    stats 가 주어지면 parquet open + row_group read 누적 시간을
    stats['t_parquet'] 에 기록.

    progress_rows: dict[shard_name -> max_submitted_row] — resume HWM.
    해당 shard 에서 row ≤ HWM 은 skip (이전 run 에서 submit 한 것으로 간주).

    Note: re_caption column 은 step 1 filter 단계에서만 필요. crop step 은
    이미 .npy index 로 row 가 결정돼 있으므로 'url' 만 읽으면 충분.
    """
    progress_rows = progress_rows or {}

    def _add_parquet(dt):
        if stats is None:
            return
        with stats['lock']:
            stats['t_parquet'] += dt
            now = time.time()
            if stats.get('parquet_first', 0.0) == 0.0:
                stats['parquet_first'] = now - dt
            if now > stats.get('parquet_last', 0.0):
                stats['parquet_last'] = now

    for shard_path in shard_paths:
        if stop_event.is_set():
            return
        shard_name = Path(shard_path).stem
        matched = _load_shard_index(out_root, shard_name)
        if matched is None or matched.size == 0:
            continue
        hwm = progress_rows.get(shard_name)
        try:
            t_open = time.time()
            f = fs.open(shard_path, 'rb')
            pf = pq.ParquetFile(f)
            _add_parquet(time.time() - t_open)
        except Exception as e:
            print(f'[skip {shard_name}] open fail: {e}', flush=True)
            continue
        try:
            offset_rg = 0
            for rg in range(pf.num_row_groups):
                if stop_event.is_set():
                    break
                rg_size = pf.metadata.row_group(rg).num_rows
                lo = int(np.searchsorted(matched, offset_rg))
                hi = int(np.searchsorted(matched, offset_rg + rg_size))
                if lo == hi:
                    offset_rg += rg_size
                    continue
                # whole-RG fast skip when every matched row is at/below HWM.
                if hwm is not None and int(matched[hi - 1]) <= hwm:
                    offset_rg += rg_size
                    continue
                t_rg = time.time()
                tbl = pf.read_row_group(rg, columns=['url'])
                _add_parquet(time.time() - t_rg)
                urls = tbl.column('url').to_pylist()
                rg_positions = (matched[lo:hi] - offset_rg).tolist()
                rg_metas = []
                for pos in rg_positions:
                    if stop_event.is_set():
                        break
                    row = offset_rg + pos
                    if hwm is not None and row <= hwm:
                        continue
                    u = urls[pos]
                    if not u:
                        continue
                    rg_metas.append({'shard': shard_name, 'row': row,
                                     'url': u})
                offset_rg += rg_size
                if rg_metas:
                    yield shard_name, rg_metas
        finally:
            f.close()


def _stream_shard_metas(shard_paths, fs, out_root, stop_event, stats=None,
                        prefetch=4, progress_rows=None):
    """Q3.2: parquet I/O 를 background thread 에서 미리 진행.
    main thread (producer) 가 fetch submit 하는 동안 다음 row_group/shard 읽기.
    """
    SENTINEL = object()
    chunk_q: Queue = Queue(maxsize=prefetch)

    def _reader():
        try:
            for item in _iter_shard_metas(
                    shard_paths, fs, out_root, stop_event, stats,
                    progress_rows=progress_rows):
                if stop_event.is_set():
                    break
                chunk_q.put(item)
        except Exception as e:
            msg = f'parquet-reader: {type(e).__name__}: {e}'
            print(f'[parquet-reader] err: {msg}', flush=True)
            # fail-fast: 정상 exhaustion 으로 위장되지 않도록 fatal flag 설정 +
            # master_stop 전파.
            if stats is not None:
                with stats['lock']:
                    if not stats.get('fatal_err'):
                        stats['fatal_err'] = msg
            stop_event.set()
        finally:
            chunk_q.put(SENTINEL)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    try:
        while True:
            item = chunk_q.get()
            if item is SENTINEL:
                return
            yield item
    finally:
        # drain
        while True:
            item = chunk_q.get(timeout=0.1) if not chunk_q.empty() else None
            if item is None or item is SENTINEL:
                break


class _PushbackIter:
    """단일-슬롯 pushback 가능한 iterator wrapper.

    Producer 가 RG 중간에 chunk_stop / fetch_cap 으로 break 할 때, 미제출 metas
    slice 를 push() 로 돌려 놓으면 다음 chunk 의 producer 가 그 시점부터 이어 받음.
    background reader 가 이미 parquet 에서 읽어버린 RG 라도 손실 없이 재사용 가능.
    """
    __slots__ = ('_base', '_buf')

    def __init__(self, base):
        self._base = iter(base)
        self._buf = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._buf is not None:
            v = self._buf
            self._buf = None
            return v
        return next(self._base)

    def push(self, value):
        if self._buf is not None:
            raise RuntimeError('_PushbackIter only supports a single pushback slot')
        self._buf = value

    def close(self):
        try:
            close = getattr(self._base, 'close', None)
            if callable(close):
                close()
        except Exception:
            pass


def _mark_shard_done(done_path, shard_name):
    with open(done_path, 'a') as df:
        df.write(shard_name + '\n')


def _load_progress_rows(progress_path):
    """resume HWM 로드. {shard_name: max_submitted_row}. 파일 없거나 깨졌으면 {}."""
    if not progress_path.exists():
        return {}
    try:
        raw = json.loads(progress_path.read_text())
        return {str(k): int(v) for k, v in raw.items()}
    except Exception as e:
        print(f'[progress] load fail ({e}), starting clean', flush=True)
        return {}


def _save_progress_rows(progress_path, rows):
    """atomic write — temp file + rename."""
    tmp = progress_path.with_suffix(progress_path.suffix + '.tmp')
    tmp.write_text(json.dumps(rows, sort_keys=True))
    tmp.replace(progress_path)


def _producer_run_threads(metas_iter, fetch_q, fetch_threads, timeout,
                          chunk_stop, master_stop, n_consumers, stats,
                          shard_state, done_path):
    """One chunk's producer. metas_iter 는 chunks 간 공유되는 generator —
    이 chunk 가 fetch_cap 만큼 submit 후 yield 끊고 다음 chunk producer 가 이어 받음.

    shard_state: {'futs': dict[shard_name -> [Future]], 'mark_threads': list,
                  'current_shard': str|None} — chunk 경계를 가로지르는 shard tracking.
    """
    pool = ThreadPoolExecutor(max_workers=fetch_threads)

    def _stopped():
        return chunk_stop.is_set() or master_stop.is_set()

    def _fetch_then_push(meta):
        if _stopped():
            return
        url = meta['url']
        t0 = time.time()
        img = fetch_image_bgr(url, timeout)
        dt = time.time() - t0
        with stats['lock']:
            stats['t_fetch'] += dt
            stats['n_fetch'] += 1
            if img is not None:
                stats['n_fetch_ok'] += 1
                stats['t_fetch_ok'] += dt
            else:
                stats['t_fetch_fail'] += dt
        if img is None or _stopped():
            return
        fetch_q.put((meta, img))

    def _wait_and_mark(futs, shard_name):
        for fut in futs:
            try:
                fut.result()
            except Exception:
                pass
        if not master_stop.is_set():
            _mark_shard_done(done_path, shard_name)

    fetch_cap = stats.get('fetch_cap', 0)
    n_submitted = 0
    submission_cap_hit = False
    gen_exhausted = True  # 기본 True; break 시 False 로
    try:
        for shard_name, metas in metas_iter:
            if master_stop.is_set():
                gen_exhausted = False
                break
            # 이전 chunk 가 끝냈을 수도 있는 shard 의 mark_done 처리는 transition 시점에.
            if shard_name != shard_state['current_shard']:
                prev = shard_state['current_shard']
                if prev is not None and prev in shard_state['futs']:
                    pf = shard_state['futs'].pop(prev)
                    t = threading.Thread(target=_wait_and_mark,
                                         args=(pf, prev), daemon=True)
                    t.start()
                    shard_state['mark_threads'].append(t)
                shard_state['current_shard'] = shard_name
                shard_state['futs'].setdefault(shard_name, [])

            # RG 안에서 row 단위 cap/stop check + pushback.
            # chunk_stop / fetch_cap 가 row i 에서 fire 하면 metas[i:] 를 push back →
            # 다음 chunk 의 producer 가 같은 shard 의 같은 RG row i 부터 이어받음.
            broke_mid_rg = False
            i = 0
            n = len(metas)
            while i < n:
                m = metas[i]
                if master_stop.is_set():
                    break
                if chunk_stop.is_set():
                    metas_iter.push((shard_name, metas[i:]))
                    broke_mid_rg = True
                    gen_exhausted = False
                    break
                if stats.get('fatal_err'):
                    metas_iter.push((shard_name, metas[i:]))
                    broke_mid_rg = True
                    gen_exhausted = False
                    break
                if fetch_cap > 0 and n_submitted >= fetch_cap:
                    metas_iter.push((shard_name, metas[i:]))
                    submission_cap_hit = True
                    broke_mid_rg = True
                    gen_exhausted = False
                    break
                shard_state['futs'][shard_name].append(
                    pool.submit(_fetch_then_push, m))
                # row HWM tracking — resume 시 이 값 이하 row 는 skip.
                row = m['row']
                cur = shard_state['max_row'].get(shard_name, -1)
                if row > cur:
                    shard_state['max_row'][shard_name] = row
                n_submitted += 1
                i += 1

            if master_stop.is_set():
                gen_exhausted = False
                break
            if broke_mid_rg:
                break  # outer loop 도 끝 — chunk 종료 / cap 도달

        with stats['lock']:
            stats['n_submitted'] += n_submitted
            if gen_exhausted:
                stats['gen_exhausted'] = True
    finally:
        pool.shutdown(wait=True)
        for _ in range(n_consumers):
            fetch_q.put(END)


_EMPTY_PRED = {'polygons': [], 'scores': []}


def _normalize_pred(preds_dict):
    """detectron2 raw output dict → {'polygons','scores'}.
    DPText-DETR 출력 attr 순회: polygons / ctrl_points / pred_polys / pred_boxes.
    """
    if preds_dict is None:
        return {'polygons': [], 'scores': []}
    instances = preds_dict.get('instances', None) if isinstance(preds_dict, dict) else None
    if instances is None or len(instances) == 0:
        return {'polygons': [], 'scores': []}
    instances = instances.to('cpu')
    scores = (instances.scores.numpy().tolist()
              if instances.has('scores') else [1.0] * len(instances))
    cands = None
    for attr in ('polygons', 'ctrl_points', 'pred_polys'):
        if instances.has(attr):
            cands = getattr(instances, attr)
            break
    polys, ss = [], []
    if cands is not None:
        try:
            cands = np.asarray(cands)
        except Exception:
            cands = list(cands)
        for poly, sc in zip(cands, scores):
            p = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            if len(p) >= 3:
                polys.append(p)
                ss.append(float(sc))
    elif instances.has('pred_boxes'):
        boxes = instances.pred_boxes.tensor.numpy()
        for box, sc in zip(boxes, scores):
            x1, y1, x2, y2 = box.tolist()
            polys.append(np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                                    dtype=np.float32))
            ss.append(float(sc))
    return {'polygons': polys, 'scores': ss}


def _predict_batch_raw(predictor, imgs_bgr, fp16=False):
    """list of HxWx3 uint8 BGR → list of model output dict (each has 'instances').

    DefaultPredictor 의 single-image __call__ 을 우회하고 model 에 list 직접 전달.
    detectron2 GeneralizedRCNN/meta_arch 가 backbone 단계에서 padding 후 batch forward 하므로
    실제 GPU throughput 이 batch dim 만큼 scale 됨 (가변 size 는 padding 비용 있음).

    fp16=True 시 torch.autocast(cuda, fp16) 으로 mixed precision inference.
    weights 는 fp32 유지 — autocast 가 op 별 fp16 kernel 자동 dispatch (deformable
    attention 같이 fp16 kernel 없는 custom op 는 fp32 fallback).
    """
    import torch
    if not imgs_bgr:
        return []
    inputs = []
    for img in imgs_bgr:
        original = img
        if predictor.input_format == 'RGB':
            original = original[:, :, ::-1]
        h, w = original.shape[:2]
        t = predictor.aug.get_transform(original).apply_image(original)
        t = torch.as_tensor(t.astype('float32').transpose(2, 0, 1))
        inputs.append({'image': t, 'height': h, 'width': w})
    with torch.no_grad():
        if fp16:
            # torch 1.9 의 cuda.amp.autocast 는 dtype 인자 없음 (torch 1.10+ 만 지원).
            # default 가 fp16 이므로 무인자로 호출.
            with torch.cuda.amp.autocast():
                return predictor.model(inputs)
        return predictor.model(inputs)


def _is_cuda_oom(exc):
    """torch.cuda.OutOfMemoryError 는 PyTorch 1.13+ 에만 존재. 1.9 등 구버전은
    plain RuntimeError 에 'out of memory' 메시지. 양쪽 다 처리."""
    import torch
    oom_cls = getattr(torch.cuda, 'OutOfMemoryError', None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    return isinstance(exc, RuntimeError) and 'out of memory' in str(exc).lower()


def _safe_predict_batch(predictor, imgs_bgr, stats, device):
    """OOM 발생 시 batch 를 절반으로 쪼개 재귀 retry. 1장에서도 OOM 이면 그 image skip.
    fp16 여부는 stats['fp16'] 로 읽음 (run_crop 에서 1회 set)."""
    import torch
    if not imgs_bgr:
        return []
    fp16 = bool(stats.get('fp16', False))
    try:
        return _predict_batch_raw(predictor, imgs_bgr, fp16=fp16)
    except Exception as e:
        if _is_cuda_oom(e):
            torch.cuda.empty_cache()
            with stats['lock']:
                stats['n_oom'] += 1
            if len(imgs_bgr) == 1:
                h, w = imgs_bgr[0].shape[:2]
                print(f'[oom] {device} skipping image {h}x{w}', flush=True)
                return [None]
            mid = len(imgs_bgr) // 2
            left = _safe_predict_batch(predictor, imgs_bgr[:mid], stats, device)
            right = _safe_predict_batch(predictor, imgs_bgr[mid:], stats, device)
            return left + right
        msg = f'gpu {device}: {type(e).__name__}: {e}'
        print(f'[gpu] err on {msg}', flush=True)
        # fail-fast: CUDA context 한 번 깨지면 같은 device 의 후속 launch 도
        # 줄줄이 실패하므로 spin 대신 fatal_err 로 main 에 신호.
        with stats['lock']:
            if not stats.get('fatal_err'):
                stats['fatal_err'] = msg
        return [None] * len(imgs_bgr)


def _gpu_consumer_run(device, fetch_q, post_q, predictors, batch_size,
                      stats, stop_event):
    """GPU forward 만 담당 — predictor.model([inputs]) 로 진짜 batched forward.
    `batch_size` 는 한 번에 GPU 에 묶어 보내는 image 수 (실제 GPU throughput 결정).
    crop / NMS / lmdb-write / preview 는 _post_worker_run 이 담당 (parallel).

    predictors: list of (name, DefaultPredictor) — 같은 device 에 로드. ArT + TT.
    pred 는 image 당 list-of-N (predictor 별 dict {'polygons','scores'}).
    """
    import torch
    if device.startswith('cuda:'):
        torch.cuda.set_device(int(device.split(':', 1)[1]))

    pending = []  # list[(meta, bgr_img)]

    def _flush_to_post():
        if not pending:
            return
        t0 = time.time()
        imgs = [img for _, img in pending]
        # per_predictor[j] = list[len(imgs)] of model output dict (or None on OOM-skip)
        per_predictor_raw = []
        for _, predictor in predictors:
            per_predictor_raw.append(
                _safe_predict_batch(predictor, imgs, stats, device))
        # transpose to per-image-per-predictor + normalize
        n = len(imgs)
        m = len(predictors)
        per_image_preds = [
            [_normalize_pred(per_predictor_raw[j][i]) for j in range(m)]
            for i in range(n)
        ]
        t1 = time.time()
        t_gpu_batch = t1 - t0
        with stats['lock']:
            stats['t_gpu'] += t_gpu_batch
            stats['n_batch'] += 1
            if stats['gpu_first'] == 0.0 or t0 < stats['gpu_first']:
                stats['gpu_first'] = t0
            if t1 > stats['gpu_last']:
                stats['gpu_last'] = t1
        post_q.put((list(pending), per_image_preds))
        pending.clear()

    while True:
        item = fetch_q.get()
        if item is END:
            _flush_to_post()
            return
        if stop_event.is_set() or stats.get('fatal_err'):
            continue  # drain 만 하고 처리 X (fatal_err 시도 join 위해 END 까지 받음)
        pending.append(item)
        if len(pending) >= batch_size:
            _flush_to_post()


def _post_worker_run(post_q, jpeg_quality, score_thr, iou_thr, max_keep,
                     lmdb_env, lmdb_lock, lmdb_state,
                     crops_dir, save_crops,
                     preview_items, preview_lock, preview_n,
                     debug_orig_dir, save_originals, stats, stop_event):
    """GPU forward 결과 (batch + per-image per-detector pred) 받아서
    raw output 합산 → AABB-NMS (IoU>iou_thr) → top-K → AABB crop → encode →
    lmdb write → preview.
    GPU consumer 와 병렬 — GPU 가 다음 batch forward 하는 동안 진행.
    """
    while True:
        item = post_q.get()
        if item is END:
            return
        batch_pending, preds = item

        local_t_crop = 0.0
        local_t_write = 0.0
        local_imgs = 0
        local_crops = 0
        per_image = []  # (meta, img, polys_per_det, consensus_boxes, kept_crops, encoded_jpgs)
        crop_phase_start = time.time()
        for (meta, img), pred_per_det in zip(batch_pending, preds):
            t1 = time.time()
            # detector 별 score 컷 + np 변환 (preview 용 polys_per_det 도 같이 빌드)
            polys_per_det = []
            combined_pool = []
            for pred in pred_per_det:
                ps = pred.get('polygons', [])
                ss = pred.get('scores', [])
                kp = []
                for poly, s in zip(ps, ss):
                    if s < score_thr:
                        continue
                    poly_np = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                    kp.append(poly_np)
                    combined_pool.append({'poly': poly_np, 'score': float(s)})
                polys_per_det.append(kp)

            # AABB-NMS combined (모든 model raw 합산 → IoU 기준 dedupe → top-K cap)
            consensus = aabb_nms_combined(combined_pool, iou_thr, max_keep)

            kept_crops = []
            encoded = []
            H, W = img.shape[:2]
            for (x1, y1, x2, y2, _sc) in consensus:
                xi1 = max(0, int(x1))
                yi1 = max(0, int(y1))
                xi2 = min(W, int(np.ceil(x2)))
                yi2 = min(H, int(np.ceil(y2)))
                if xi2 <= xi1 or yi2 <= yi1:
                    continue
                crop = img[yi1:yi2, xi1:xi2]
                if crop.size == 0:
                    continue
                ok, buf = cv2.imencode('.jpg', crop,
                                       [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                if not ok:
                    continue
                kept_crops.append(crop)
                encoded.append(buf.tobytes())
            local_t_crop += time.time() - t1
            per_image.append((meta, img, polys_per_det, consensus, kept_crops, encoded))
            local_imgs += 1
            local_crops += len(kept_crops)

        crop_phase_end = time.time()
        t2 = crop_phase_end
        batch_total = sum(len(e) for _, _, _, _, _, e in per_image)
        # Chunked mode: target_crops 는 *이번 chunk* 가 추가하기로 한 양.
        # chunk_start_idx 는 chunk 시작 시점의 lmdb_state['idx'] — 이전 chunk 들 기여 누적.
        target_crops = stats.get('target_crops', 0) or 0
        chunk_start_idx = stats.get('chunk_start_idx', 0)
        actually_written = 0
        if batch_total:
            with lmdb_lock:
                with lmdb_env.begin(write=True) as txn:
                    done = False
                    for _, _, _, _, _, encoded in per_image:
                        if done:
                            break
                        for jpg_bytes in encoded:
                            if (target_crops
                                    and (lmdb_state['idx'] - chunk_start_idx)
                                        >= target_crops):
                                done = True
                                break
                            lmdb_state['idx'] += 1
                            actually_written += 1
                            txn.put(f'image-{lmdb_state["idx"]:09d}'.encode(),
                                    jpg_bytes)
                    txn.put(b'num-samples', str(lmdb_state['idx']).encode())
        if save_crops:
            for meta, _, _, _, kept_crops, _ in per_image:
                for ci, crop in enumerate(kept_crops):
                    key = f"{meta['shard']}__r{meta['row']:08d}__c{ci:02d}"
                    cv2.imwrite(str(crops_dir / f'{key}.jpg'),
                                crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        write_phase_end = time.time()
        local_t_write = write_phase_end - t2
        wrote_anything = bool(batch_total) or bool(save_crops and per_image)

        for meta, img, polys_per_det, consensus, kept_crops, _ in per_image:
            if preview_n > 0 and kept_crops:
                with preview_lock:
                    if len(preview_items) < preview_n:
                        orig_name = f"{meta['shard']}__r{meta['row']:08d}.jpg"
                        if save_originals:
                            cv2.imwrite(str(debug_orig_dir / orig_name),
                                        img, [cv2.IMWRITE_JPEG_QUALITY, 88])
                        vis = overlay_ensemble(img, polys_per_det, consensus)
                        preview_items.append({
                            'name': orig_name,
                            'vis_b64': encode_jpeg(vis),
                            'crops': [{'score': box[4], 'b64': encode_jpeg(c)}
                                      for box, c in zip(consensus, kept_crops)],
                        })

        # target_crops trim 적용 시 actually_written 이 local_crops 보다 작을 수 있음.
        # batch 가 lmdb 안 쓰면 (batch_total=0) actually_written=0 이라 local_crops 그대로 사용.
        crops_to_record = actually_written if batch_total else local_crops
        with stats['lock']:
            stats['t_crop'] += local_t_crop
            stats['t_write'] += local_t_write
            stats['imgs'] += local_imgs
            stats['crops'] += crops_to_record
            if local_imgs:
                if stats['crop_first'] == 0.0 or crop_phase_start < stats['crop_first']:
                    stats['crop_first'] = crop_phase_start
                if crop_phase_end > stats['crop_last']:
                    stats['crop_last'] = crop_phase_end
            if wrote_anything:
                if stats['write_first'] == 0.0 or t2 < stats['write_first']:
                    stats['write_first'] = t2
                if write_phase_end > stats['write_last']:
                    stats['write_last'] = write_phase_end
            stats['pbar'].update(local_imgs)
            stats['pbar'].set_postfix({'crops': stats['crops']})

        if stats['target'] and stats['imgs'] >= stats['target']:
            stop_event.set()
        if target_crops and (lmdb_state['idx'] - chunk_start_idx) >= target_crops:
            stop_event.set()


def _render_crops_preview(items, out_dir) -> str:
    rows = []
    for it in items:
        crops_html = ''.join(
            f'<div class="crop">'
            f'  <img src="data:image/jpeg;base64,{c["b64"]}"/>'
            f'  <div class="sc">conf {c["score"]:.2f}</div>'
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
<title>RDC-U crops preview</title>
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
<h1>RDC-U crops · {len(rows)} preview images, originals at <code>{html.escape(str(out_dir))}</code></h1>
{''.join(rows)}
</body></html>
"""


def _short_count(n: int) -> str:
    """1_000_000 → '1M', 100_000 → '100K', 12_345 → '12345'. 0 → '0'."""
    if n == 0:
        return '0'
    for unit, s in ((1_000_000_000, 'B'), (1_000_000, 'M'), (1_000, 'K')):
        if n >= unit and n % unit == 0:
            return f'{n // unit}{s}'
    return str(n)


def _crop_run_suffix(args) -> str:
    """LMDB / crops / done-marker 폴더 구분용 suffix.

    - target_crops > 0 → `_tN`  (N = short form, 예: 1M, 100K)
    - max_keep < 100 → `_kN`   (max_keep=100 은 default 라 생략 — backwards compat)
    - shard_mod > 1 → `_wRR`   (cluster split)
    - fp16 → `_fp16`           (mixed precision inference)
    """
    parts = []
    if args.target_crops > 0:
        parts.append(f't{_short_count(args.target_crops)}')
    if 0 < args.max_keep < 100:
        parts.append(f'k{args.max_keep}')
    if args.shard_mod > 1:
        parts.append(f'w{args.shard_rem:02d}')
    if getattr(args, 'fp16', False):
        parts.append('fp16')
    return '_' + '_'.join(parts) if parts else ''


def run_crop(args):
    total_t0 = time.time()
    import torch
    suffix = _crop_run_suffix(args)

    crops_dir = args.out_root / f'crops{suffix}'
    if args.save_crops:
        crops_dir.mkdir(parents=True, exist_ok=True)

    debug_orig_dir = args.out_root / 'debug' / 'originals'
    if args.crop_preview_n > 0 and args.save_preview_images:
        debug_orig_dir.mkdir(parents=True, exist_ok=True)

    lmdb_dir = args.out_root / f'lmdb{suffix}'
    lmdb_dir.mkdir(parents=True, exist_ok=True)
    done_path = args.out_root / f'_done_shards{suffix}.txt'

    done_set = set()
    if done_path.exists():
        done_set = {l.strip() for l in done_path.read_text().splitlines() if l.strip()}
    progress_path = args.out_root / f'_progress_rows{suffix}.json'
    progress_state = _load_progress_rows(progress_path)
    # done shards 는 progress_state 에서 제거 (불필요).
    for s in list(progress_state.keys()):
        if s in done_set:
            progress_state.pop(s)

    fs = HfFileSystem()
    paths = sorted(fs.glob(f'datasets/{args.dataset}/data/train_data/*.parquet'))
    if args.shard_mod > 1:
        paths = [p for p in paths
                 if shard_idx_of(Path(p).stem) % args.shard_mod == args.shard_rem]
    indexed_before_done, missing_before_done = _filter_paths_with_index(
        paths, args.out_root)
    done_indexed = [p for p in indexed_before_done if Path(p).stem in done_set]
    paths = [p for p in indexed_before_done if Path(p).stem not in done_set]
    if args.max_shards:
        paths = paths[:args.max_shards]

    n_gpus = args.num_gpus
    if torch.cuda.is_available():
        if n_gpus <= 0:
            n_gpus = max(1, torch.cuda.device_count())
        devices = [f'cuda:{i}' for i in range(n_gpus)]
    else:
        devices = ['cpu']
    n_consumers = len(devices)

    print(f'[crop] dataset    : {args.dataset}')
    print(f'[crop] shards     : {len(paths)} (machine {args.shard_rem}/{args.shard_mod}, '
          f'{len(indexed_before_done)} indexed, {len(done_indexed)} done-indexed, '
          f'{missing_before_done} no-index)')
    print(f'[crop] out_root   : {args.out_root}')
    print(f'[crop] lmdb       : {lmdb_dir.name} (Union14M-U format, no label key)')
    if args.save_crops:
        print(f'[crop] crops dir  : {crops_dir.name}/ (loose jpg dump)')
    print(f'[crop] target     : {args.target or "no cap"} imgs')
    # target_crops 가 있으면 fetch upper-bound 산출 — over-fetch 방지.
    fetch_cap = 0
    if args.target_crops > 0:
        ey = max(args.expected_yield, 0.1)
        ed = max(args.expected_decode_rate, 0.05)
        sf = max(args.fetch_safety, 1.0)
        imgs_needed = max(1, int(np.ceil(args.target_crops / ey)))
        urls_needed = max(1, int(np.ceil(imgs_needed / ed)))
        fetch_cap = max(1, int(np.ceil(urls_needed * sf)))
        print(f'[crop] target_crops: {args.target_crops:,} → ~{imgs_needed:,} imgs, '
              f'fetch cap {fetch_cap:,} '
              f'(yield={ey}, decode={ed}, safety={sf}x)')
    print(f'[crop] devices    : {devices}')
    print(f'[crop] batch_size : {args.batch_size} per GPU')
    print(f'[crop] fetch      : {args.fetch_threads} threads, '
          f'queue {args.queue_size}, timeout connect={args.timeout[0]}s read={args.timeout[1]}s')
    print(f'[crop] filter     : per-detector score>={args.score_thr}, '
          f'AABB-NMS IoU>{args.iou_thr}, top-{args.max_keep} per image')
    if args.crop_preview_n > 0:
        if args.save_preview_images:
            print(f'[crop] crop_preview_n : {args.crop_preview_n} '
                  f'(debug originals → {debug_orig_dir})')
        else:
            print(f'[crop] crop_preview_n : {args.crop_preview_n} '
                  f'(html only; --save_preview_images to dump originals)')

    if not paths:
        reason = 'nothing to do'
        if missing_before_done and not indexed_before_done:
            print(f'[crop] no index at {args.out_root}/index/. '
                  f'run --steps 1 first (or just `python rdc-u.py`).')
            reason = 'no filter index'
        elif done_indexed:
            done_names = ', '.join(Path(p).stem for p in done_indexed[:5])
            more = ' ...' if len(done_indexed) > 5 else ''
            print(f'[crop] no indexed shards left: {len(done_indexed)} indexed '
                  f'shard(s) are already marked done in {done_path} '
                  f'({done_names}{more}).')
            print('[crop] to rerun crop for the same shard, use a fresh --out_root '
                  'or remove the done marker file.')
            reason = 'all indexed shards already marked done'
        else:
            print('[crop] nothing to do')
        wall = time.time() - total_t0
        return {
            'step': 'crop',
            'wall': wall,
            'setup_wall': wall,
            'load_wall': 0.0,
            'pipeline_wall': 0.0,
            'skipped': True,
            'reason': reason,
        }

    model_names = [n for n, _, _ in DPTEXT_MODELS]
    print(f'[crop] loading DPText-DETR models {model_names} on each device ...',
          flush=True)
    load_t0 = time.time()
    if not DPTEXT_REPO.exists():
        raise RuntimeError(f'[crop] DPText-DETR repo not found at {DPTEXT_REPO}. '
                           f'Clone: git clone https://github.com/ymy-k/DPText-DETR.git '
                           f'into {DPTEXT_REPO.parent}')
    sys.path.insert(0, str(DPTEXT_REPO))
    from detectron2.engine import DefaultPredictor
    try:
        from adet.config import get_cfg as _get_cfg  # type: ignore
    except ImportError:
        from detectron2.config import get_cfg as _get_cfg  # type: ignore

    # fp16: MSDeformAttn custom CUDA op (_C.ms_deform_attn_forward) 은 fp16 kernel
    # 미구현이라 autocast 가 fp16 입력 넣으면 RuntimeError. custom_fwd(cast_inputs=fp32)
    # 로 wrap → autocast 안에서도 이 op 만 fp32 입력으로 호출. 나머지 (ResNet50 등) 는
    # fp16 dispatch 유지.
    if getattr(args, 'fp16', False):
        import torch as _torch
        from adet.layers.ms_deform_attn import _MSDeformAttnFunction
        _orig_msda_fwd = _MSDeformAttnFunction.forward
        if hasattr(_orig_msda_fwd, '__func__'):
            _orig_msda_fwd = _orig_msda_fwd.__func__

        @staticmethod
        @_torch.cuda.amp.custom_fwd(cast_inputs=_torch.float32)
        def _fp32_msda_fwd(ctx, *fwd_args):
            return _orig_msda_fwd(ctx, *fwd_args)
        _MSDeformAttnFunction.forward = _fp32_msda_fwd
        print('[crop] fp16: MSDeformAttn._MSDeformAttnFunction.forward wrapped '
              'with custom_fwd(cast_inputs=fp32)')

    predictors_per_dev = []  # list[device_idx] -> list[(name, predictor)]
    for dev in devices:
        dev_preds = []
        for name, cfg_rel, w_rel in DPTEXT_MODELS:
            cfg_path = DPTEXT_REPO / cfg_rel
            w_path = RDC_ROOT / w_rel
            if not cfg_path.exists() or not w_path.exists():
                raise RuntimeError(
                    f'[crop] missing for {name}: cfg={cfg_path}  weights={w_path}')
            cfg = _get_cfg()
            cfg.merge_from_file(str(cfg_path))
            cfg.MODEL.WEIGHTS = str(w_path)
            cfg.MODEL.DEVICE = dev
            if hasattr(cfg.MODEL, 'TRANSFORMER') and \
                    hasattr(cfg.MODEL.TRANSFORMER, 'INFERENCE_TH_TEST'):
                cfg.MODEL.TRANSFORMER.INFERENCE_TH_TEST = args.score_thr
            predictor = DefaultPredictor(cfg)
            dev_preds.append((name, predictor))
            print(f'  {dev}: {name} ready', flush=True)
        predictors_per_dev.append(dev_preds)

    # warmup 제거됨 — multi-GPU + DPText-DETR (deformable attention 커스텀 CUDA op)
    # 환경에서 main thread 에서 device 별 sequential warmup 시 illegal memory access.
    # 첫 batch 가 cudnn JIT 비용 ~1-2s 흡수하지만 full run 기준 무시할 수준.
    # _gpu_consumer_run 은 thread 별로 torch.cuda.set_device() 하므로 안전.
    load_wall = time.time() - load_t0

    pbar = tqdm(desc='crop', unit='img', smoothing=0.05)
    stats = {
        'lock': threading.Lock(),
        'pbar': pbar,
        'fp16': bool(getattr(args, 'fp16', False)),
        'imgs': 0, 'crops': 0,
        'target': args.target, 'target_crops': args.target_crops,
        'fetch_cap': fetch_cap, 'n_submitted': 0, 'n_skipped_by_cap': 0,
        'chunk_start_idx': 0,
        'gen_exhausted': False,
        'fatal_err': None,
        't_fetch': 0.0, 't_fetch_ok': 0.0, 't_fetch_fail': 0.0,
        't_gpu': 0.0, 't_crop': 0.0, 't_write': 0.0,
        't_parquet': 0.0,
        'n_fetch': 0, 'n_fetch_ok': 0, 'n_batch': 0, 'n_oom': 0,
        # phase wall span: min(first_t)/max(last_t) across workers; 0=never ran
        'gpu_first': 0.0, 'gpu_last': 0.0,
        'crop_first': 0.0, 'crop_last': 0.0,
        'write_first': 0.0, 'write_last': 0.0,
        'parquet_first': 0.0, 'parquet_last': 0.0,
    }
    preview_items = []
    preview_lock = threading.Lock()

    lmdb_env = lmdb.open(str(lmdb_dir), map_size=args.lmdb_map_size,
                         subdir=True, readonly=False, lock=True,
                         meminit=False, map_async=True)
    lmdb_lock = threading.Lock()
    with lmdb_env.begin() as _t:
        existing = _t.get(b'num-samples')
    lmdb_state = {'idx': int(existing) if existing else 0}
    if lmdb_state['idx']:
        print(f'[crop] lmdb resume : {lmdb_state["idx"]} existing samples')

    # Chunked pipeline: 모델/lmdb/metas_iter 는 persistent. threads 는 chunk 마다 spawn-and-join.
    # _PushbackIter 로 mid-RG break 시 unsubmitted metas slice 를 다음 chunk 가 이어받게 함.
    master_stop = threading.Event()
    shard_state = {'futs': {}, 'mark_threads': [], 'current_shard': None,
                   'max_row': {}}
    metas_iter = _PushbackIter(
        _stream_shard_metas(paths, fs, args.out_root, master_stop, stats,
                            progress_rows=progress_state))
    if progress_state:
        print(f'[crop] row HWM resume: {len(progress_state)} partial shard(s)'
              f' (skip rows ≤ HWM)', flush=True)

    fetch_wall_total = 0.0
    chunk_idx = 0
    crops_at_run_start = lmdb_state['idx']
    # display 용 endpoint — 이번 run 끝나면 lmdb 가 도달할 위치.
    # break/chunk 로직은 여전히 args.target_crops (per-run) 기준 — 여기는 표시만.
    crops_target_endpoint = (crops_at_run_start + args.target_crops
                             if args.target_crops > 0 else 0)

    t0 = time.time()
    try:
        while True:
            if master_stop.is_set():
                break
            if stats.get('fatal_err'):
                break
            if stats.get('gen_exhausted'):
                break
            crops_so_far = lmdb_state['idx'] - crops_at_run_start
            if args.target_crops > 0 and crops_so_far >= args.target_crops:
                break
            if args.target > 0 and stats['imgs'] >= args.target:
                break

            chunk_idx += 1

            # 이 chunk 의 target_crops + fetch_cap 산정.
            if args.chunk_crops > 0:
                if args.target_crops > 0:
                    remaining = args.target_crops - crops_so_far
                    chunk_tc = min(args.chunk_crops, remaining)
                else:
                    chunk_tc = args.chunk_crops
            else:
                chunk_tc = args.target_crops  # 0 = no cap, single-chunk legacy 동작

            if chunk_tc > 0:
                ey = max(args.expected_yield, 0.1)
                ed = max(args.expected_decode_rate, 0.05)
                sf = max(args.fetch_safety, 1.0)
                chunk_fc = max(1, int(np.ceil(chunk_tc / ey / ed * sf)))
            else:
                chunk_fc = 0

            with stats['lock']:
                stats['target_crops'] = chunk_tc
                stats['fetch_cap'] = chunk_fc
                stats['chunk_start_idx'] = lmdb_state['idx']

            if args.chunk_crops > 0:
                # progress 는 lmdb 누적 (기존 + 이번 run) / endpoint 로 표시.
                tot_label = (f'/{crops_target_endpoint:,}'
                             if crops_target_endpoint else '')
                print(f'\n[crop] chunk {chunk_idx}: target +{chunk_tc:,} crops, '
                      f'fetch cap {chunk_fc:,} '
                      f'(progress {lmdb_state["idx"]:,}{tot_label})', flush=True)

            chunk_stop = threading.Event()
            chunk_fetch_q: Queue = Queue(maxsize=args.queue_size)
            chunk_post_q: Queue = Queue(maxsize=2 * n_consumers)
            chunk_lmdb_idx_at_start = lmdb_state['idx']

            post_workers = []
            for _ in range(n_consumers):
                t = threading.Thread(
                    target=_post_worker_run,
                    args=(chunk_post_q, args.jpeg_quality,
                          args.score_thr, args.iou_thr, args.max_keep,
                          lmdb_env, lmdb_lock, lmdb_state,
                          crops_dir, args.save_crops,
                          preview_items, preview_lock, args.crop_preview_n,
                          debug_orig_dir, args.save_preview_images,
                          stats, chunk_stop),
                    daemon=True)
                t.start()
                post_workers.append(t)

            consumers = []
            for dev, dev_preds in zip(devices, predictors_per_dev):
                t = threading.Thread(
                    target=_gpu_consumer_run,
                    args=(dev, chunk_fetch_q, chunk_post_q, dev_preds,
                          args.batch_size, stats, chunk_stop),
                    daemon=True)
                t.start()
                consumers.append(t)

            prod = threading.Thread(
                target=_producer_run_threads,
                args=(metas_iter, chunk_fetch_q, args.fetch_threads, args.timeout,
                      chunk_stop, master_stop, n_consumers, stats,
                      shard_state, done_path),
                daemon=True)
            fetch_wall_t0 = time.time()
            prod.start()
            prod.join()
            fetch_wall_total += time.time() - fetch_wall_t0
            for t in consumers:
                t.join()
            for _ in range(n_consumers):
                chunk_post_q.put(END)
            for t in post_workers:
                t.join()

            chunk_added = lmdb_state['idx'] - chunk_lmdb_idx_at_start
            if args.chunk_crops > 0:
                # total 도 lmdb 누적 (기존 포함) / endpoint 로 표시.
                tot_label = (f'/{crops_target_endpoint:,}'
                             if crops_target_endpoint else '')
                print(f'[crop] chunk {chunk_idx} done: +{chunk_added:,} crops '
                      f'(total {lmdb_state["idx"]:,}{tot_label})', flush=True)

            # Row HWM checkpoint persist — fully-done shards 는 done_path 가 진실,
            # 진행 중 shard 만 progress_state 에 남김.
            done_set_now = set()
            if done_path.exists():
                done_set_now = {l.strip() for l
                                in done_path.read_text().splitlines() if l.strip()}
            for s, mr in shard_state.get('max_row', {}).items():
                if s in done_set_now:
                    progress_state.pop(s, None)
                else:
                    progress_state[s] = max(progress_state.get(s, -1), mr)
            try:
                _save_progress_rows(progress_path, progress_state)
            except Exception as e:
                print(f'[progress] save fail (non-fatal): {e}', flush=True)

            if args.chunk_crops <= 0:
                break  # single-chunk legacy mode
            if chunk_added == 0:
                # 0 crop chunk — yield 0 + URL 다 죽음. 3회 연속이면 stop (무한 loop 방지).
                stats['_zero_streak'] = stats.get('_zero_streak', 0) + 1
                if stats['_zero_streak'] >= 3:
                    print('[crop] 3 chunks in a row produced 0 crops, stopping.',
                          flush=True)
                    break
            else:
                stats['_zero_streak'] = 0

        # metas_iter 가 자연 소진됐으면 마지막 shard 의 futures 도 다 처리됐을 것 — 마크.
        if (stats.get('gen_exhausted') and shard_state['current_shard'] is not None
                and shard_state['current_shard'] in shard_state['futs']):
            last = shard_state['current_shard']
            futs = shard_state['futs'].pop(last)
            for f in futs:
                try:
                    f.result()
                except Exception:
                    pass
            _mark_shard_done(done_path, last)
        for t in shard_state.get('mark_threads', []):
            t.join(timeout=10.0)
    finally:
        master_stop.set()
        try:
            metas_iter.close()
        except Exception:
            pass
        pbar.close()
        lmdb_env.sync()
        lmdb_env.close()
        # 마지막 progress persist — fatal_err / interrupt 시에도 resume 가능하게.
        try:
            done_set_final = set()
            if done_path.exists():
                done_set_final = {l.strip() for l
                                  in done_path.read_text().splitlines() if l.strip()}
            for s, mr in shard_state.get('max_row', {}).items():
                if s in done_set_final:
                    progress_state.pop(s, None)
                else:
                    progress_state[s] = max(progress_state.get(s, -1), mr)
            _save_progress_rows(progress_path, progress_state)
        except Exception as e:
            print(f'[progress] final save fail (non-fatal): {e}', flush=True)

    if stats.get('fatal_err'):
        raise RuntimeError(f"crop aborted: {stats['fatal_err']}")

    # Legacy variable 호환 — 후속 summary print 에서 fetch_wall 사용.
    fetch_wall = fetch_wall_total

    process_wall = time.time() - t0
    total_wall = time.time() - total_t0
    setup_wall = max(0.0, total_wall - load_wall - process_wall)
    fetch_n = max(stats['n_fetch'], 1)
    fetch_ok = stats['n_fetch_ok']
    processed_imgs = stats['imgs']
    fetched_not_processed = max(0, fetch_ok - processed_imgs)
    avg_fetch_ms = stats['t_fetch'] / fetch_n * 1000
    fail_n = max(0, stats['n_fetch'] - fetch_ok)
    avg_fetch_ok_ms = (stats['t_fetch_ok'] / fetch_ok * 1000) if fetch_ok else 0.0
    avg_fetch_fail_ms = (stats['t_fetch_fail'] / fail_n * 1000) if fail_n else 0.0
    avg_batch_ms = stats['t_gpu'] / max(stats['n_batch'], 1) * 1000
    crops_per_img = stats['crops'] / max(processed_imgs, 1)
    gpu_wall = (stats['gpu_last'] - stats['gpu_first']) if stats['gpu_first'] else 0.0
    crop_wall = (stats['crop_last'] - stats['crop_first']) if stats['crop_first'] else 0.0
    write_wall = (stats['write_last'] - stats['write_first']) if stats['write_first'] else 0.0
    target_note = ''
    if args.target and stats['imgs'] >= args.target:
        target_note = f' (target {args.target} imgs reached; batch/parallel overshoot)'

    print()
    print(f'[crop] total wall : {fmt_secs(total_wall)}')
    print(f'[crop] setup/load : {fmt_secs(setup_wall)} setup + '
          f'{fmt_secs(load_wall)} DPText-DETR (2 models) load')
    print(f'[crop] pipeline   : {fmt_secs(process_wall)} '
          f'(fetch + GPU + crop/write overlapped)')
    print(f'[crop] fetch urls : {stats["n_fetch"]} tried, {fetch_ok} decoded '
          f'({100*fetch_ok/fetch_n:.1f}%)')
    if fetched_not_processed:
        print(f'[crop] skipped    : {fetched_not_processed} decoded images '
              f'left unprocessed after target/backlog stop')
    print(f'[crop] gpu imgs   : {processed_imgs} processed by DPText-DETR{target_note}')
    print(f'[crop] crops      : {stats["crops"]} '
          f'({crops_per_img:.2f} per gpu img)')
    oom_note = f' · {stats["n_oom"]} OOM-retries' if stats['n_oom'] else ''
    gpu_idle = max(0.0, process_wall - gpu_wall)
    gpu_util_active = stats['t_gpu'] / max(gpu_wall * n_consumers, 1e-9) * 100
    gpu_util_pipeline = stats['t_gpu'] / max(process_wall * n_consumers, 1e-9) * 100
    parquet_wall = (stats['parquet_last'] - stats['parquet_first']) \
        if stats['parquet_first'] else 0.0
    print(f'[crop] phase timing (wall = envelope of activity; worker-sum = busy time):')
    print(f'  parquet: wall {fmt_secs(parquet_wall)} · '
          f'worker-sum {fmt_secs(stats["t_parquet"])} '
          f'(producer 단일 thread; shard footer + matched row_group 다운로드)')
    print(f'  fetch  : wall {fmt_secs(fetch_wall)} · '
          f'worker-sum {fmt_secs(stats["t_fetch"])} '
          f'({fetch_n} reqs · ok avg {avg_fetch_ok_ms:.0f}ms · '
          f'fail avg {avg_fetch_fail_ms:.0f}ms)')
    print(f'  gpu    : wall {fmt_secs(gpu_wall)} · '
          f'worker-sum {fmt_secs(stats["t_gpu"])} '
          f'({stats["n_batch"]} batches × {n_consumers} GPUs · '
          f'avg {avg_batch_ms:.0f}ms/batch{oom_note})')
    print(f'           util {gpu_util_active:.0f}% during active wall · '
          f'{gpu_util_pipeline:.0f}% over pipeline · '
          f'idle {fmt_secs(gpu_idle)} (waiting on fetch backlog)')
    print(f'  crop   : wall {fmt_secs(crop_wall)} · '
          f'worker-sum {fmt_secs(stats["t_crop"])}')
    print(f'  write  : wall {fmt_secs(write_wall)} · '
          f'worker-sum {fmt_secs(stats["t_write"])}')
    bottleneck = 'fetch' if fetch_wall >= gpu_wall - 1 else 'gpu'
    print(f'[crop] bottleneck: {bottleneck} '
          f'(pipeline wall {fmt_secs(process_wall)}, '
          f'gpu wall {fmt_secs(gpu_wall)}, fetch wall {fmt_secs(fetch_wall)})')

    # per-image steady-state cost — warmup/target-stop tail 와 무관한 본질 비율.
    # fetch 는 decoded image 기준 (실패한 URL 들도 cumulative 에 포함, ok+fail 다 합산).
    # gpu 는 processed image 기준 (single-GPU latency × predictor 들).
    fetch_per_img = stats['t_fetch'] / max(fetch_ok, 1)
    gpu_per_img_single = stats['t_gpu'] / max(processed_imgs, 1)
    gpu_per_img_parallel = gpu_per_img_single / max(n_consumers, 1)
    fetch_eff_threads = stats['t_fetch'] / max(fetch_wall, 1e-9)
    fetch_max_throughput = fetch_eff_threads / max(fetch_per_img, 1e-9)
    gpu_max_throughput = n_consumers / max(gpu_per_img_single, 1e-9)
    print(f'[crop] per-image cost (steady-state, warmup/tail 무관):')
    print(f'  fetch  : {fetch_per_img*1000:.0f}ms/img cumulative '
          f'(over all URL tries · fail 포함)')
    print(f'  gpu    : {gpu_per_img_single*1000:.0f}ms/img per-GPU (2 predictors) · '
          f'{gpu_per_img_parallel*1000:.0f}ms/img across {n_consumers} GPUs')
    print(f'  ratio  : fetch/GPU = {fetch_per_img/max(gpu_per_img_parallel, 1e-9):.1f}× '
          f'(per-image fetch cost vs available GPU time)')
    print(f'[crop] max sustainable throughput by phase:')
    print(f'  fetch  : {fetch_max_throughput:.1f} img/s '
          f'(effective {fetch_eff_threads:.0f} concurrent / 128 fetch_threads)')
    print(f'  gpu    : {gpu_max_throughput:.1f} img/s '
          f'({n_consumers} GPUs × {1/max(gpu_per_img_single, 1e-9):.1f} img/s/GPU)')

    preview_html = None
    if args.crop_preview_n > 0 and preview_items:
        html_path = args.out_root / f'crops_preview{suffix}.html'
        html_path.write_text(
            _render_crops_preview(preview_items, debug_orig_dir), encoding='utf-8')
        print(f'[crop] preview : {html_path}')
        preview_html = str(html_path)
    return {
        'step': 'crop',
        'wall': total_wall,
        'setup_wall': setup_wall,
        'load_wall': load_wall,
        'pipeline_wall': process_wall,
        'fetch_wall': fetch_wall,
        'gpu_wall': gpu_wall,
        'crop_wall': crop_wall,
        'write_wall': write_wall,
        'parquet_wall': parquet_wall,
        'fetch_sum': stats['t_fetch'],
        'gpu_sum': stats['t_gpu'],
        'crop_sum': stats['t_crop'],
        'write_sum': stats['t_write'],
        'parquet_sum': stats['t_parquet'],
        'urls_tried': stats['n_fetch'],
        'urls_decoded': fetch_ok,
        'decoded_not_processed': fetched_not_processed,
        'gpu_imgs': processed_imgs,
        'crops': stats['crops'],
        'batches': stats['n_batch'],
        'gpus': n_consumers,
        'target': args.target,
        'preview_html': preview_html,
    }


# ═════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════

def _write_run_summary(args, total_wall, summaries):
    """전체 실행 통계를 out_root/SUMMARY{suffix}.json 으로 저장.
    LMDB/crops 와 동일한 suffix (예: _t10K_k5) 를 써서 setting 별로 분리.
    args 직렬화 시 Path/tuple 등은 default=str 로 fallback."""
    payload = {
        'dataset': getattr(args, 'dataset', None),
        'steps_requested': args.steps,
        'total_wall_seconds': round(total_wall, 1),
        'args': vars(args),
        'steps': {s.get('step', f'step_{i}'): s
                  for i, s in enumerate(summaries)},
    }
    suffix = _crop_run_suffix(args)
    out_path = args.out_root / f'SUMMARY{suffix}.json'
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    print(f'[summary] wrote {out_path}')


def _print_final_summary(total_wall, summaries):
    print('\n========== [summary] ==========')
    stage_wall = sum(s.get('wall', 0.0) for s in summaries)
    other_wall = max(0.0, total_wall - stage_wall)

    print(f'[total] wall {fmt_secs(total_wall)}')
    print(f'  startup/other before/around steps : {fmt_secs(other_wall)}')
    print(f'  requested step wall sum           : {fmt_secs(stage_wall)}')

    for s in summaries:
        step = s.get('step', 'unknown')
        if step == 'filter':
            preview = s.get('preview') or {}
            scan_wall = s.get('scan_wall', 0.0)
            preview_wall = preview.get('wall', 0.0)
            filter_other = max(0.0, s.get('wall', 0.0) - scan_wall - preview_wall)
            rows = s.get('rows')
            rows_txt = f'{rows:,}' if rows is not None else 'unknown/cached'
            match_rate = s.get('match_rate')
            rate_txt = f' ({100 * match_rate:.2f}%)' if match_rate is not None else ''
            cached_txt = ''
            if s.get('cached'):
                cached_txt = f', cached shards {s.get("cached", 0)}/{s.get("shards", 0)}'

            print()
            print(f'[step 1: filter] wall {fmt_secs(s.get("wall", 0.0))}')
            print(f'  1. setup/cache bookkeeping : {fmt_secs(filter_other)}')
            print(f'  2. text scan/index         : {fmt_secs(scan_wall)}')
            print(f'       rows {rows_txt}{cached_txt}')
            print(f'       matched {s.get("matched", 0):,}{rate_txt}')
            print(f'  3. filter preview          : {fmt_secs(preview_wall)}')
            if preview.get('target', 0):
                print(f'       candidate collect wall  : '
                      f'{fmt_secs(preview.get("collect_wall", 0.0))}')
                print(f'       URL fetch wall          : '
                      f'{fmt_secs(preview.get("fetch_wall", 0.0))} '
                      f'({preview.get("attempts", 0)} urls, '
                      f'{preview.get("threads", 0)} threads)')
                print(f'       saved preview images    : '
                      f'{preview.get("saved", 0)}/{preview.get("target", 0)}')
                if preview.get('html'):
                    print(f'       preview html            : {preview.get("html")}')
            else:
                print('       disabled')
        elif step == 'crop':
            print()
            if s.get('skipped'):
                print(f'[step 2: crop] wall {fmt_secs(s.get("wall", 0.0))}')
                print(f'  skipped: {s.get("reason", "unknown")}')
                continue

            backlog = s.get('decoded_not_processed', 0)
            print(f'[step 2: crop] wall {fmt_secs(s.get("wall", 0.0))}')
            print(f'  1. setup/shard bookkeeping : '
                  f'{fmt_secs(s.get("setup_wall", 0.0))}')
            print(f'  2. DPText-DETR load (2 mdl): '
                  f'{fmt_secs(s.get("load_wall", 0.0))}')
            print(f'  3. pipeline wall           : '
                  f'{fmt_secs(s.get("pipeline_wall", 0.0))}')
            pipeline_w = s.get('pipeline_wall', 0.0)
            gpu_w = s.get('gpu_wall', 0.0)
            fetch_w = s.get('fetch_wall', 0.0)
            n_g = max(s.get('gpus', 1), 1)
            util_active = s.get('gpu_sum', 0.0) / max(gpu_w * n_g, 1e-9) * 100
            util_pipe = s.get('gpu_sum', 0.0) / max(pipeline_w * n_g, 1e-9) * 100
            gpu_idle = max(0.0, pipeline_w - gpu_w)
            print(f'       phase timing (wall = envelope; worker-sum = busy):')
            print(f'         parquet: wall {fmt_secs(s.get("parquet_wall", 0.0))} · '
                  f'worker-sum {fmt_secs(s.get("parquet_sum", 0.0))} '
                  f'(footer + matched row_group)')
            print(f'         fetch  : wall {fmt_secs(fetch_w)} · '
                  f'worker-sum {fmt_secs(s.get("fetch_sum", 0.0))} '
                  f'({s.get("urls_tried", 0)} tried, '
                  f'{s.get("urls_decoded", 0)} decoded)')
            print(f'         gpu    : wall {fmt_secs(gpu_w)} · '
                  f'worker-sum {fmt_secs(s.get("gpu_sum", 0.0))} '
                  f'({s.get("batches", 0)} batches × {n_g} GPUs · '
                  f'util {util_active:.0f}% active / {util_pipe:.0f}% pipeline · '
                  f'idle {fmt_secs(gpu_idle)})')
            print(f'         crop   : wall {fmt_secs(s.get("crop_wall", 0.0))} · '
                  f'worker-sum {fmt_secs(s.get("crop_sum", 0.0))}')
            print(f'         write  : wall {fmt_secs(s.get("write_wall", 0.0))} · '
                  f'worker-sum {fmt_secs(s.get("write_sum", 0.0))}')
            bottleneck = 'fetch' if fetch_w >= gpu_w - 1 else 'gpu'
            print(f'       bottleneck              : {bottleneck}')
            print(f'       GPU input images        : {s.get("gpu_imgs", 0)}')
            print(f'       saved crops             : {s.get("crops", 0)}')
            if backlog:
                print(f'       decoded backlog skipped : {backlog}')
            if s.get('preview_html'):
                print(f'       crop preview html       : {s.get("preview_html")}')


def _remove_path(p: Path) -> bool:
    if not p.exists() and not p.is_symlink():
        return False
    if p.is_symlink() or p.is_file():
        p.unlink()
    elif p.is_dir():
        shutil.rmtree(p)
    return True


def _fresh_start(args, requested):
    targets = []
    if 'filter' in requested:
        targets.extend([
            args.out_root / 'index',
            args.out_root / 'filter_preview',
            args.out_root / 'filter_preview.html',
        ])
    if 'crop' in requested:
        # NOTE: --fresh 는 *현재 args 의 suffix* 에 해당하는 exp 만 지움 —
        # 다른 (target_crops, max_keep, shard_mod) 조합의 LMDB 는 손대지 않음.
        targets.extend([
            args.out_root / 'debug' / 'originals',
            args.out_root / 'crops_preview.html',
        ])
        suffix = _crop_run_suffix(args)
        targets.append(args.out_root / f'crops{suffix}')
        targets.append(args.out_root / f'lmdb{suffix}')
        targets.append(args.out_root / f'_done_shards{suffix}.txt')
        targets.append(args.out_root / f'_progress_rows{suffix}.json')

    seen = set()
    removed = []
    for p in targets:
        p = p.resolve()
        if p in seen:
            continue
        seen.add(p)
        if _remove_path(p):
            removed.append(p)

    if removed:
        print(f'[fresh] removed {len(removed)} existing output path(s):')
        for p in removed:
            print(f'  - {p}')
    else:
        print('[fresh] no existing outputs to remove for requested steps')


VALID_STEPS = {'filter', 'crop'}
STEP_ALIAS = {'1': 'filter', '2': 'crop', 'index': 'filter'}


def _parse_timeout(s):
    """'X' or 'X,Y' → (connect, read) tuple."""
    parts = [p.strip() for p in str(s).split(',')]
    try:
        if len(parts) == 1:
            v = float(parts[0])
            return (v, v)
        if len(parts) == 2:
            return (float(parts[0]), float(parts[1]))
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(
        f"timeout must be 'X' or 'X,Y' (got {s!r})")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--steps', default='1,2',
                    help='comma-separated stages — name 또는 번호 사용 가능: '
                         '1=filter, 2=crop. default filter,crop (1,2). '
                         '예: --steps 1,2 또는 --steps filter,crop')
    ap.add_argument('--out_root', type=expand, default=expand('./out_rdc-u'))
    ap.add_argument('--dataset', default=DEFAULT_DS)

    # 풀 실행 shortcut — default 는 항상 small crop 모드
    ap.add_argument('--full', action='store_true',
                    help='편의: --max_shards 0 --target 0 으로 override (전체 처리). '
                         '디폴트는 small crop 모드 (max_shards=1, target=100).')
    ap.add_argument('--fresh', action='store_true',
                    help='delete existing outputs for requested steps before running. '
                         '--steps 2 keeps filter index and resets crop outputs; '
                         '--steps 1,2 resets both filter and crop outputs.')
    ap.add_argument('--debug', action='store_true',
                    help='dev iteration mode — out_root 의 basename 을 ./DEBUG/ 아래로 '
                         '재배치. 실 산출물 디렉터리 오염 방지. 예: '
                         '--out_root ./out_rdc-u --debug → ./DEBUG/out_rdc-u/.')

    # filter
    ap.add_argument('--workers', type=int, default=8,
                    help='[filter] parallel shard scan threads')
    ap.add_argument('--max_shards', type=int, default=None,
                    help='[filter/crop] limit shards (0 = all). 명시 안 하면: '
                         '--full 일 때 0, --target_crops 있을 때 그 값 기반 자동 산정 '
                         '(baseline URLs × 2 / ~71K matches/shard 으로 cushion 포함), '
                         '둘 다 없을 때 1 (small dev).')
    ap.add_argument('--force', action='store_true',
                    help='[filter] ignore existing .npy index cache')

    # filter preview
    ap.add_argument('--filter_preview_n', '--n', dest='filter_preview_n',
                    type=int, default=10,
                    help='[filter] filtered image/text preview count '
                         '(0 = disable). --n is kept as a short alias.')
    ap.add_argument('--filter_preview_threads', type=int, default=32,
                    help='[filter-preview] parallel URL fetch threads')
    ap.add_argument('--seed', type=int, default=0,
                    help='[filter-preview] shuffle seed')
    ap.add_argument('--max_attempts', type=int, default=200,
                    help='[filter-preview] dead-URL safety cap')

    # crop — DPText-DETR ArT + TT (configs/weights 는 DPTEXT_MODELS 상수 참조)
    ap.add_argument('--num_gpus', type=int, default=0,
                    help='[crop] STD GPU 수 (0 = auto-detect)')
    ap.add_argument('--score_thr', type=float, default=0.3,
                    help='[crop] per-model score threshold (DPText-DETR '
                         'INFERENCE_TH_TEST 도 같은 값으로 set).')
    ap.add_argument('--iou_thr', type=float, default=0.5,
                    help='[crop] AABB-NMS IoU threshold (combined NMS 후).')
    ap.add_argument('--max_keep', type=int, default=100,
                    help='[crop] image 당 최종 detection top-K cap. 0 이면 cap 없음.')
    ap.add_argument('--jpeg_quality', type=int, default=90)
    ap.add_argument('--batch_size', type=int, default=2,
                    help='[crop] GPU forward batch 크기 — predictor.model([N inputs]) '
                         '로 실제 batched forward. DPText-DETR 은 ResizeShortestEdge(1200, 1900) '
                         '으로 input 을 키우는 DETR-family 라 transformer encoder cost 가 '
                         'B×HW 로 linear 하게 늘어 batch 효과가 작음. 24GB 3090 / 가변 size '
                         '워크로드 측정 결과 bs=2 가 sweet spot (uniform 720p 239ms/img · '
                         'mixed 223ms/img). bs↑ 시 padding 비용으로 mixed-size throughput 오히려 '
                         '하락. OOM 시 자동 bisect retry.')
    ap.add_argument('--fetch_threads', type=int, default=128,
                    help='[crop] 동시 URL fetch thread 수.')
    ap.add_argument('--queue_size', type=int, default=512,
                    help='[crop] producer→consumer queue 크기. fetch_threads 의 4배 권장.')
    ap.add_argument('--timeout', type=_parse_timeout, default=(3.0, 5.0),
                    help='[filter-preview/crop] URL fetch timeout (초). '
                         '"X" 는 connect+read 양쪽에 X 적용, '
                         '"X,Y" 는 connect=X · read=Y. default 3,5.')
    ap.add_argument('--target', type=int, default=None,
                    help='[crop] stop after N GPU-processed images '
                         '(0 = no cap). 명시 안 하면 --target_crops 가 있을 때 0, '
                         '없을 때 100 (small dev). crop 수가 아니라 GPU 입력 이미지 수.')
    ap.add_argument('--target_crops', type=int, default=0,
                    help='[crop] stop after N crops written to lmdb (0 = no cap). '
                         '--target 와 함께 쓰면 먼저 도달하는 쪽이 stop trigger. '
                         '이 값으로 fetch upper-bound 도 자동 산출 — '
                         'ceil(target_crops / expected_yield / expected_decode_rate '
                         '* fetch_safety) 만큼만 producer 가 submit (과도 fetch 방지).')
    ap.add_argument('--expected_yield', type=float, default=None,
                    help='[crop] target_crops fetch-cap 산출용 예상 crops/img. '
                         '명시 안 하면 min(12, max_keep) — max_keep 으로 yield 가 cap 되므로 '
                         '자동 연동. probe 측정 12.13 (cap=100 기준).')
    ap.add_argument('--expected_decode_rate', type=float, default=0.5,
                    help='[crop] target_crops fetch-cap 산출용 예상 fetch ok rate. '
                         'probe 측정 0.751 이지만 small target / dead URL ratio 변동 '
                         '커서 보수적으로 0.5 (target 미달 방지 우선).')
    ap.add_argument('--fetch_safety', type=float, default=2.5,
                    help='[crop] fetch upper-bound safety multiplier — yield/decode 가 '
                         '예상보다 나빠도 target 도달하도록. 1.0 이면 hard cap. '
                         'default 2.5 = decode/yield 변동 + producer/consumer race '
                         '모두 흡수. 더 줄이면 fetch 적게 시도 → target 미달 risk.')
    ap.add_argument('--chunk_crops', type=int, default=None,
                    help='[crop] chunked processing — 한 chunk 당 추가할 crop 수. '
                         '명시 안 하면 --target_crops 기반 자동 산정 '
                         '(target≤100K = 0 단일 pass; 그 외 clip(target/30, 50K, 1M)). '
                         '0 = 단일 pass. 모델/lmdb/metas iterator 는 chunks 간 persistent, '
                         'threads 만 spawn-and-join. RG 중간에 끊겨도 pushback 으로 손실 없음.')
    ap.add_argument('--shard_mod', type=int, default=1,
                    help='[crop] cluster split: total worker count')
    ap.add_argument('--shard_rem', type=int, default=0,
                    help='[crop] cluster split: this worker idx')
    ap.add_argument('--crop_preview_n', '--preview_n', dest='crop_preview_n',
                    type=int, default=20,
                    help='[crop] render polygon-overlay HTML for N source images '
                         '(0 = disable). --preview_n is kept as an alias.')
    ap.add_argument('--save_preview_images', action='store_true',
                    help='[filter/crop] also dump preview originals to disk '
                         '(filter_preview/*.jpg and debug/originals/*.jpg). '
                         'HTML embeds images as base64 so disk dump is optional.')
    ap.add_argument('--save_crops', action='store_true',
                    help='[crop] also dump individual crop jpgs to crops/. '
                         'default 는 LMDB 만 저장하고 loose jpg 는 안 떨굼.')
    ap.add_argument('--lmdb_map_size', type=int, default=1 << 40,
                    help='[crop] LMDB env map_size (sparse alloc on Linux). '
                         'default 1TB — 부족하면 키워야 OS-level error 안 남.')
    ap.add_argument('--fp16', action='store_true',
                    help='[crop] torch.autocast(cuda, fp16) 로 mixed precision '
                         'inference. weights 는 fp32 유지하고 autocast 가 op별 fp16 '
                         'kernel 자동 dispatch. deformable attention 같이 fp16 kernel '
                         '없는 custom op 는 fp32 fallback. output (lmdb/crops_preview/'
                         'SUMMARY) 모두 _fp16 suffix 로 분리되어 fp32 와 비교 가능.')

    args = ap.parse_args()

    # --target sentinel resolve: target_crops 있으면 default 0 (간섭 방지), 없으면 100.
    if args.target is None:
        args.target = 0 if args.target_crops > 0 else 100

    # --expected_yield sentinel resolve: max_keep 으로 cap 되므로 둘 중 작은 값.
    # max_keep=0 (no cap) 이면 probe 측정값 12.0 그대로.
    if args.expected_yield is None:
        if args.max_keep > 0:
            args.expected_yield = float(min(12.0, args.max_keep))
        else:
            args.expected_yield = 12.0

    # --max_shards sentinel resolve: target_crops 있으면 자동 산정, 없으면 1.
    # --full 은 아래에서 0 으로 override.
    if args.max_shards is None:
        if args.target_crops > 0 and not args.full:
            ey = max(args.expected_yield, 0.1)
            ed = max(args.expected_decode_rate, 0.05)
            sf = max(args.fetch_safety, 1.0)
            matches_per_shard = 71_000  # probe 측정값 (매우 안정적, std 0.09%)
            urls_needed = int(np.ceil(args.target_crops / ey / ed * sf))
            # 2x cushion: shard 간 yield 변동 + producer 가 step 2 도중 부족 안 겪게.
            args.max_shards = max(1, int(np.ceil(urls_needed * 2 / matches_per_shard)))
            print(f'[setup] auto --max_shards={args.max_shards} '
                  f'(target_crops={args.target_crops:,} → '
                  f'~{matches_per_shard * args.max_shards:,} matched URLs available, '
                  f'baseline need ~{urls_needed:,})', flush=True)
        else:
            args.max_shards = 1

    # --chunk_crops sentinel resolve: target_crops 기반 chunks 수 ~30 목표 (50K-1M clip).
    # 작은 target 은 chunking overhead 가 메모리 이득보다 커서 단일 pass.
    if args.chunk_crops is None:
        if args.target_crops <= 100_000:
            args.chunk_crops = 0
        else:
            args.chunk_crops = max(50_000, min(1_000_000, args.target_crops // 30))
            n_chunks = int(np.ceil(args.target_crops / args.chunk_crops))
            print(f'[setup] auto --chunk_crops={args.chunk_crops:,} '
                  f'(~{n_chunks} chunks)', flush=True)

    if args.full:
        args.max_shards = 0
        args.target = 0

    # --debug: out_root 를 ./DEBUG/<basename>/ 으로 옮김. dev 와 실 산출물 분리.
    if args.debug:
        debug_root = (Path.cwd() / 'DEBUG').resolve()
        args.out_root = debug_root / args.out_root.name
        print(f'[setup] --debug → out_root = {args.out_root}', flush=True)

    requested = [STEP_ALIAS.get(s.strip(), s.strip())
                 for s in args.steps.split(',') if s.strip()]
    bad = [s for s in requested if s not in VALID_STEPS]
    if bad:
        ap.error(f'unknown steps: {bad}. valid: {sorted(VALID_STEPS)} '
                 f'or aliases {sorted(STEP_ALIAS)}')
    if not requested:
        ap.error('--steps must include at least one stage')

    # crop step 은 detectron2 + AdelaiDet (DPText-DETR) env 가 필요.
    # 잘못된 env 에서 실행 시 step 1 까지 다 돌고 step 2 직전에 fail 하는 걸 방지.
    if 'crop' in requested:
        missing = []
        try:
            import detectron2  # noqa: F401
        except ImportError:
            missing.append('detectron2')
        try:
            sys.path.insert(0, str(DPTEXT_REPO))
            import adet  # noqa: F401
        except ImportError:
            missing.append('adet (DPText-DETR)')
        if missing:
            print()
            print('=' * 70)
            print(f'[ERROR] crop step 에 필요한 모듈 없음: {", ".join(missing)}')
            print('=' * 70)
            print('rdc-u.py 는 `detectron2` conda env 에서 실행해야 합니다.')
            print('현재 python:', sys.executable)
            print()
            print('올바른 실행 명령:')
            print(f'  /data/isaackang/anaconda3/envs/detectron2/bin/python {sys.argv[0]} '
                  + ' '.join(sys.argv[1:]))
            print()
            print('또는 env 활성화 후:')
            print('  conda activate detectron2')
            print(f'  python {sys.argv[0]} ' + ' '.join(sys.argv[1:]))
            print('=' * 70)
            sys.exit(2)

    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        _fresh_start(args, requested)

    runners = {'filter': run_index, 'crop': run_crop}
    summaries = []
    for step in requested:
        print(f'\n========== [step: {step}] ==========')
        step_t0 = time.time()
        summary = runners[step](args)
        if summary is None:
            summary = {'step': step, 'wall': time.time() - step_t0}
        summary.setdefault('step', step)
        summary.setdefault('wall', time.time() - step_t0)
        summaries.append(summary)

    total_wall = time.time() - SCRIPT_T0
    _print_final_summary(total_wall, summaries)
    _write_run_summary(args, total_wall, summaries)


if __name__ == '__main__':
    main()
