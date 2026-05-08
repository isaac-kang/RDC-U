"""
Recap-DataComp-1B 의 모든 train shard 에 strict text filter 를 적용,
match 된 row 의 (shard, row) index 만 shard 별 .npy 로 저장.

출력:
  <out_dir>/<shard_basename>.npy   : uint32 배열 (matched row indices)
  <out_dir>/SUMMARY.json           : 진행/총합 요약

특징:
  - 이미 처리된 shard 는 건너뜀 (resume 가능)
  - ThreadPoolExecutor 로 I/O 병렬 (디폴트 8 worker)
  - filter 로직은 sample_recap_datacomp.caption_has_str_signal 그대로 재사용

사용법:
    python build_str_index.py                       # 디폴트 8 workers, all shards
    python build_str_index.py --workers 16
    python build_str_index.py --max_shards 50       # 작게 시작해서 시험
    python build_str_index.py --force               # 기존 .npy 무시하고 재계산
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from tqdm import tqdm

# 같은 디렉토리의 strict filter 재사용
sys.path.insert(0, str(Path(__file__).parent))
from sample_recap_datacomp import caption_has_str_signal  # noqa: E402


DEFAULT_DS = 'UCSC-VLAA/Recap-DataComp-1B'


def expand(p: str) -> Path:
    return Path(p).expanduser().resolve()


def process_shard(fs: HfFileSystem, shard_path: str, out_dir: Path,
                  force: bool) -> dict:
    """shard 1개 처리. 매치 row index 저장하고 통계 반환."""
    name = Path(shard_path).stem  # train-00000-of-04627
    out_file = out_dir / f'{name}.npy'

    if out_file.exists() and not force:
        arr = np.load(out_file)
        return {'shard': name, 'rows': None, 'matches': int(arr.size),
                'elapsed': 0.0, 'cached': True}

    t0 = time.time()
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
    return {'shard': name, 'rows': offset, 'matches': int(arr.size),
            'elapsed': time.time() - t0, 'cached': False}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default=DEFAULT_DS)
    ap.add_argument('--out_dir', type=expand, default=expand('./recap_str_index'),
                    help='shard 별 .npy 저장 경로')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max_shards', type=int, default=None,
                    help='None 이면 전체 (4627 train shards). 작게 시험할 때 지정.')
    ap.add_argument('--force', action='store_true',
                    help='기존 .npy 가 있어도 다시 계산')
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    fs = HfFileSystem()
    paths = sorted(fs.glob(f'datasets/{args.dataset}/data/train_data/*.parquet'))
    if args.max_shards:
        paths = paths[:args.max_shards]

    print(f'dataset : {args.dataset}')
    print(f'shards  : {len(paths)}')
    print(f'workers : {args.workers}')
    print(f'out_dir : {args.out_dir}')
    print()

    t0 = time.time()
    results = []
    pbar = tqdm(total=len(paths), desc='shards', smoothing=0.05)
    matched_total = 0
    rows_total = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_shard, fs, p, args.out_dir, args.force)
                   for p in paths]
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            matched_total += r['matches']
            if r['rows'] is not None:
                rows_total += r['rows']
            pbar.set_postfix({'matched': f'{matched_total/1e6:.1f}M'})
            pbar.update(1)
    pbar.close()

    wall = time.time() - t0
    summary = {
        'dataset': args.dataset,
        'shards_processed': len(results),
        'shards_total': len(paths),
        'rows_scanned': rows_total,
        'matched_total': matched_total,
        'match_rate': (matched_total / rows_total) if rows_total else None,
        'wall_seconds': round(wall, 1),
        'workers': args.workers,
    }
    (args.out_dir / 'SUMMARY.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print()
    print(f'wall      : {wall:.1f}s ({wall/3600:.2f}h)')
    print(f'rows      : {rows_total:,}')
    print(f'matched   : {matched_total:,} '
          f'({100*matched_total/max(rows_total,1):.2f}%)')
    print(f'summary   : {args.out_dir / "SUMMARY.json"}')


if __name__ == '__main__':
    main()
