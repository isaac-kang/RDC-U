"""
HuggingFace 의 UCSC-VLAA/Recap-DataComp-1B 에서 N장 샘플을 받아 이미지 + caption HTML preview 생성.

Recap-DataComp-1B 는 이미지 자체가 아니라 (url, re_caption, org_caption) 형태로 배포돼서
URL 에서 직접 다운로드해야 함. 일부 URL 은 죽어있을 수 있으므로 N 채워질 때까지 계속 시도.

NOTE: `datasets` 라이브러리는 이 데이터셋의 등록 schema 와 실제 parquet 컬럼이 안 맞아서
CastError 가 남 (re_caption_condition_diverse_topk 컬럼 누락 이슈). 그래서 huggingface_hub
+ pyarrow 로 parquet 을 직접 stream 함.

준비:
    pip install huggingface_hub pyarrow pillow requests tqdm

사용법:
    python sample_recap_datacomp.py                  # 디폴트 (10장)
    python sample_recap_datacomp.py --n 50 --seed 7
    python sample_recap_datacomp.py --out ./other_dir --no_html
"""

import argparse
import base64
import html
import io
import random
import re
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm


DEFAULT_DS = 'UCSC-VLAA/Recap-DataComp-1B'
UA = 'Mozilla/5.0 (compatible; RecapPreview/1.0)'
WANTED_COLS = ['url', 're_caption', 'org_caption']

# 인용부호 안의 string — 이미지 속 실제 텍스트일 가능성 높음 (LLaVA 패턴).
# - double-quote: body 에 apostrophe 자유 허용 ("don't stop" 통째로 잡힘)
# - single-quote: 양쪽이 word char 사이면 contraction (Don't, you're, O'Brien)
#   으로 보고 delimiter 취급 안 함. typographic quote (U+2018/9/201C/D) 도 처리.
QUOTED_RE = re.compile(
    r'[\"“]([^\"“”\n]+?)[\"”]'
    r"|"
    r"(?<!\w)['‘]([^'‘’\n]+?)['’](?!\w)"
)

# tier-1 텍스트 신호 — quoted 없어도 strong. 'context', 'textile' 같은 건 \b 가 막음.
TIER1_RE = re.compile(
    r"\b("
    r"text|word|words|"
    r"reads|says|displays|"
    r"written|titled|labeled|labelled|spelling|"
    r"inscription|inscribed|engraved"
    r")\b",
    re.IGNORECASE,
)

# strict 필터: tier-1 동사 OR 인용부호 안 string
def caption_has_str_signal(cap: str) -> bool:
    if not cap:
        return False
    return bool(TIER1_RE.search(cap) or QUOTED_RE.search(cap))


def _escape_with_highlight(text: str) -> str:
    """re_caption 의 인용부호 안 텍스트와 tier-1 동사를 highlight. HTML escape 동시 처리."""
    spans = []  # list of (start, end, css_class)
    for m in QUOTED_RE.finditer(text):
        spans.append((m.start(), m.end(), 'q'))
    for m in TIER1_RE.finditer(text):
        spans.append((m.start(), m.end(), 'v'))
    if not spans:
        return html.escape(text)
    spans.sort()
    # 겹침 제거 (앞엣게 우선)
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


def render_html(items, ds_name: str, out_dir: Path) -> str:
    """items: list of dict with keys: name, fp (Path), url, re_caption, org_caption."""
    rows = []
    for i, it in enumerate(items):
        b = it['fp'].read_bytes()
        b64 = base64.b64encode(b).decode()
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
<html lang="ko">
<head>
<meta charset="utf-8" />
<title>Recap-DataComp-1B sample · {html.escape(ds_name)}</title>
<style>
  :root {{ --col-idx: 40px; --col-img: 320px; --gap: 24px; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 32px auto; padding: 0 24px;
          background: #fff; color: #222; max-width: 1280px; }}
  header {{ margin-bottom: 28px; }}
  h1 {{ font-size: 18px; margin: 0 0 4px; font-weight: 600; }}
  .topmeta {{ color: #888; font-size: 13px; }}
  .row {{ display: grid;
          grid-template-columns: var(--col-idx) var(--col-img) 1fr;
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
</style>
</head>
<body>
<header>
  <h1>Recap-DataComp-1B sample · <code>{html.escape(ds_name)}</code></h1>
  <div class="topmeta">{len(rows)} images · saved to <code>{html.escape(str(out_dir))}</code></div>
</header>
{"".join(rows)}
</body>
</html>
"""


def expand(p: str) -> Path:
    return Path(p).expanduser().resolve()


def fetch_image(url: str, timeout: float) -> Image.Image | None:
    try:
        resp = requests.get(url, timeout=timeout, headers={'User-Agent': UA}, stream=True)
        resp.raise_for_status()
        ctype = resp.headers.get('Content-Type', '')
        if 'image' not in ctype and not url.lower().split('?')[0].endswith(
            ('.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp')
        ):
            return None
        img = Image.open(io.BytesIO(resp.content)).convert('RGB')
        return img
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', type=str, default=DEFAULT_DS,
                    help=f'HF dataset name (default: {DEFAULT_DS})')
    ap.add_argument('--split', type=str, default='train')
    ap.add_argument('--n', type=int, default=10, help='샘플 장수 (default 10)')
    ap.add_argument('--seed', type=int, default=0, help='shuffle seed')
    ap.add_argument('--shuffle_buffer', type=int, default=2000,
                    help='streaming shuffle buffer (default 2000). 0 이면 shuffle 안 함.')
    ap.add_argument('--timeout', type=float, default=8.0, help='이미지 다운로드 timeout(초)')
    ap.add_argument('--max_attempts', type=int, default=200,
                    help='URL 죽은 거 많을 수 있어서 *다운로드 시도* 상한. 도달 시 중단.')
    ap.add_argument('--filter', choices=['none', 'str'], default='none',
                    help="'str' 이면 re_caption 에 STR 신호(인용부호 안 string OR "
                         "tier-1 동사: reads/says/written/titled/labeled/the text/...) "
                         "가 있는 row 만 후보로 둠. 캡션 거른 row 는 max_attempts 에 안 셈.")
    ap.add_argument('--out', type=expand, default=expand('./recap_datacomp_samples'),
                    help='저장 폴더')
    ap.add_argument('--no_html', action='store_true', help='HTML preview 안 만들기')
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f'dataset : {args.dataset} ({args.split})')
    print(f'n       : {args.n}')
    print(f'out     : {args.out}')

    rng = random.Random(args.seed)

    # 1) parquet shard 목록 (HfFileSystem 으로 repo 내 파일 listing)
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    fs = HfFileSystem()
    repo_root = f'datasets/{args.dataset}'
    # data/*.parquet 형태 (데이터셋마다 폴더 구조 약간 다름) — recursive glob 으로 안전하게
    parquet_paths = [p for p in fs.glob(f'{repo_root}/**/*.parquet')]
    if not parquet_paths:
        print(f'[error] no parquet files under {repo_root}')
        return
    print(f'shards  : {len(parquet_paths)} parquet files')

    # 2) 다양성을 위해 shard 도 셔플
    rng.shuffle(parquet_paths)

    items = []
    pbar = tqdm(total=args.n, desc='sample')
    attempts = 0

    def _process(row):
        url = row.get('url')
        if not url:
            return None
        img = fetch_image(url, args.timeout)
        if img is None:
            return None
        idx = len(items)
        name = f'recap_{idx:05d}.jpg'
        fp = args.out / name
        img.save(fp, format='JPEG', quality=92)
        return {
            'name': name,
            'fp': fp,
            'url': url,
            're_caption': str(row.get('re_caption') or ''),
            'org_caption': str(row.get('org_caption') or ''),
        }

    # 3) shard 하나씩 열어 batch 단위로 row 뽑아 처리
    for shard_idx, shard_path in enumerate(parquet_paths):
        if len(items) >= args.n or attempts >= args.max_attempts:
            break
        try:
            f = fs.open(shard_path, 'rb')
            pf = pq.ParquetFile(f)
        except Exception as e:
            print(f'[skip shard {shard_idx}] open fail: {e}')
            continue
        if shard_idx == 0:
            print(f'fields  : {pf.schema_arrow.names}')

        # row group 도 셔플하면 같은 shard 내 다양성 ↑
        rg_indices = list(range(pf.num_row_groups))
        rng.shuffle(rg_indices)
        try:
            for rg in rg_indices:
                if len(items) >= args.n or attempts >= args.max_attempts:
                    break
                tbl = pf.read_row_group(rg, columns=WANTED_COLS)
                rows = tbl.to_pylist()
                rng.shuffle(rows)
                for row in rows:
                    if len(items) >= args.n or attempts >= args.max_attempts:
                        break
                    if args.filter == 'str' and not caption_has_str_signal(
                        row.get('re_caption') or ''
                    ):
                        continue  # 캡션 거른 row 는 attempt 안 셈
                    attempts += 1
                    rec = _process(row)
                    if rec is not None:
                        items.append(rec)
                        pbar.update(1)
        finally:
            f.close()
    pbar.close()

    print(f'\nsaved {len(items)}/{args.n} images (after {attempts} attempts) -> {args.out}')

    if len(items) == 0:
        print('[error] no images downloaded — URL 들이 다 죽었거나 네트워크 문제일 수 있음')
        return

    if not args.no_html:
        html_path = args.out.parent / f'{args.out.name}_preview.html'
        html_path.write_text(render_html(items, args.dataset, args.out),
                             encoding='utf-8')
        print(f'wrote {html_path}')


if __name__ == '__main__':
    main()
