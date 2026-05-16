# RDC-U — Recap-DataComp 에서 mining 하는 unlabeled STR crop 데이터셋

전체 흐름: parquet `re_caption` 의 regex 필터 → URL fetch → DPText-DETR 2-model ensemble (ArT / Total-Text, R50) → raw polygon 합산 → AABB-NMS → axis-aligned bbox crop.

명명 규칙: U14M-L (labeled) / U14M-U (unlabeled) 와 동일 의미의 "-U" — 글자 transcription 라벨은 없는 unlabeled crop 모음.

---

## 통합 framework — `rdc-u.py`

`--steps` 로 stage 를 골라/조합해 실행. public step 은 두 개만 둔다.

| stage | 번호 | 역할 | 산출물 |
|---|---|---|---|
| `filter` | `1` | parquet text column만 보고 caption STR filter 적용. 매치 row index를 `.npy` 로 저장하고, 일부 filtered row는 이미지+텍스트 HTML로 검수 | `<out>/index/<shard>.npy`, `<out>/index/SUMMARY.json`, `<out>/filter_preview.html` |
| `crop`   | `2` | filter index에 걸린 row만 URL fetch → DPText-DETR ArT+TotalText forward → detector raw output 합산 → AABB-NMS → bbox crop. crop preview HTML 저장 | `<out>/lmdb*/`, `<out>/crops*/<key>.jpg` (opt-in), `<out>/crops_preview*.html` |

**의존성**: `crop` 은 같은 `--out_root` 안의 `index/<shard>.npy` 를 읽어 매치 row 만 sparse 접근. index 없는 shard 는 자동 skip. `crop` 단독 실행 (`--steps 2`) 전에 step 1 을 한번 돌려야 함. default `python rdc-u.py` 는 1→2 순서로 돌아 crop 에 필요한 index 를 자동 생성함.

**기본 동작 = small crop 모드.** default 로 `--steps` 미지정 시 filter+crop(1,2), `--max_shards 1`, `--filter_preview_n 10`, `--target 100`. 풀 실행하려면 `--full` 또는 `--max_shards 0 --target 0` 명시.

`--target_crops` 를 주고 `--max_shards` 를 생략하면 `max_keep`, `expected_yield`, `expected_decode_rate`, `fetch_safety` 로 필요한 shard 수를 자동 산정한다. 이 자동 `max_shards` 값은 `--steps 1,2` 실행에서 step 1 filter와 step 2 crop 양쪽에 동일하게 적용된다.

`--steps` 는 이름 또는 번호 사용 가능. 숫자 alias 는 `1=filter`, `2=crop`. `index` 는 `filter` 의 legacy 이름으로 허용.

```bash
# 기본: text filter + filter preview, 그 다음 crop + crop preview
python rdc-u.py

# 단일 stage
python rdc-u.py --steps 1            # filter index + filter preview
python rdc-u.py --steps 2            # 1 shard 안에서 100 crop 채우면 stop
python rdc-u.py --steps 1,2          # filter + crop 작게
python rdc-u.py --fresh              # 기존 filter/crop 산출물 지우고 처음부터
python rdc-u.py --fresh --steps 2    # index 는 유지하고 crop 산출물만 새로

# 풀 실행
python rdc-u.py --full                                     # 전체 데이터 filter + crop
python rdc-u.py --steps 2 --max_shards 0 --target 100000   # cap 만 지정
```

자주 쓰는 flag:

| flag | step | 의미 |
|---|---|---|
| `--out_root` | 모두 | 출력 루트 (default `./out_rdc-u`) |
| `--full` | 모두 | shortcut: `--max_shards 0 --target 0` (전체 처리) |
| `--fresh` | 모두 | 요청한 step의 기존 산출물을 지우고 시작. `--steps 2`와 쓰면 index는 유지하고 crop만 reset |
| `--max_shards N` | filter, crop | shard 상한 (0 = all). default **1** |
| `--workers N` | filter | shard 동시 scan 스레드 |
| `--filter_preview_n N` | filter | filtered 이미지+텍스트 HTML preview 장수. default **10**, 0이면 disable |
| `--filter_preview_threads N` | filter | filter preview URL fetch 병렬 수. default **32** |
| `--num_gpus N` | crop | ensemble inference GPU 수 (0 = `torch.cuda.device_count()`) |
| `--target N` | crop | ensemble 입력 N장 처리하면 stop (0 = no cap). default **100** |
| `--target_crops N` | crop | LMDB가 N개 crop에 도달하면 stop (absolute, 기존 crop 포함). Resume 시 동일 명령어 그대로 → 부족분만 추가 생성. `--target` 이미지 cap에 먼저 도달한 경우를 제외하고 미달이면 summary 작성 후 non-zero exit |
| `--expected_yield` | crop | `--target_crops` 기반 shard/fetch 자동 산정용 crops/img. default `min(12, 0.5*max_keep)`, `max_keep=0`이면 12 |
| `--score_thr` | crop | per-detector polygon score 컷 (default **0.3**) |
| `--iou_thr` | crop | AABB-NMS IoU 임계값 (default **0.5**) |
| `--max_keep` | crop | image당 NMS 후 keep 수. 초과 시 `--seed`와 source row로 deterministic random sample |
| `--batch_size` | crop | per-GPU ensemble batch (default **2**) |
| `--fetch_threads` | crop | URL fetch ThreadPool 크기 (default 128) |
| `--shard_mod / --shard_rem` | crop | 멀티 머신 분산용 shard 분할 |
| `--crop_preview_n N` | crop | 개발용: N 장은 원본 + polygon overlay HTML 저장 (default 20) |
| `--seed N` | filter, crop | filter preview shuffle과 crop max_keep sampling을 재현 가능하게 고정 |

### 출력 디렉토리 구조

```
<out_root>/
├── index/                                # step:filter index cache
│   ├── train-00000-of-04627.npy
│   └── SUMMARY.json
├── filter_preview/                       # step:filter preview images
│   └── recap_00000.jpg ...
├── filter_preview.html                   # step:filter preview HTML
├── crops*/                               # step:crop loose jpg dump (--save_crops)
│   └── train-00042-of-04627__r00012345__c00.jpg ...
├── debug/originals/                      # step:crop, --crop_preview_n>0
│   └── train-00042-of-04627__r00012345.jpg ...
├── crops_preview*.html                   # step:crop, --crop_preview_n>0
├── lmdb*/                                # step:crop (메인 산출물, Union14M-U format)
├── _done_shards*.txt                     # step:crop resume marker
└── _progress_rows*.json                  # partial shard HWM resume marker
```

### 단계별 timing

실행 마지막에 `[summary]` 블록이 한 번 더 출력된다. 작업 순서대로 `total → step 1 filter → filter preview → step 2 crop → crop pipeline` 을 보여주고, 각 항목은 실제 elapsed `wall` 과 병렬 작업 누적 `worker-sum` 을 분리해서 적는다.

`worker-sum` 은 병렬 fetch thread나 여러 GPU에서 동시에 돈 시간을 더한 값이라 실제 wall time보다 클 수 있다. 실제 기다린 시간은 각 줄의 `wall` 을 보면 된다.

---

## 환경

- `create_env_and_download_weights.sh` — conda env `detectron2` 에 torch1.9 + detectron2 v0.6 + DPText-DETR/AdelaiDet 설치
- 외부 의존: `local-models/DPText-DETR/`, `Union14M/tools/`
- detector weights (`Union14M/checkpoints/`):
  - `dptext_art_final.pth` — DPText-DETR ArT
  - `dptext_totaltext.pth` — DPText-DETR Total-Text
