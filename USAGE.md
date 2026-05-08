# RDC-U — Recap-DataComp 에서 mining 하는 unlabeled STR crop 데이터셋

전체 흐름: parquet `re_caption` 의 regex 필터 → URL fetch → DBNet++ STD → rotate-crop.

명명 규칙: U14M-L (labeled) / U14M-U (unlabeled) 와 동일한 의미의 "-U" — 글자 transcription 라벨은 없는 unlabeled crop 모음.

---

## 1. caption 기반 후보 sample 추출 — `sample_recap_datacomp.py`

```bash
# 빠른 검수용: filter 없이 10장
python sample_recap_datacomp.py

# STR 신호 (인용부호 / tier-1 동사) 만 필터해서 N장
python sample_recap_datacomp.py --filter str --n 100 --out ./recap_str_samples

# 다른 seed/timeout
python sample_recap_datacomp.py --filter str --n 200 --seed 7 --timeout 12
```

산출물:
- `<out>/recap_NNNNN.jpg` — 다운로드된 이미지
- `<out>_preview.html` — caption highlight 된 시각 검수 페이지

자주 쓰는 flag:
- `--filter str` — `caption_has_str_signal()` 통과한 row 만 (caption 거른 건 `max_attempts` 에 안 셈)
- `--shuffle_buffer` — 0 이면 shuffle 안 함 (decode 디버깅용)
- `--no_html` — preview HTML skip
- `--max_attempts` — URL 죽은 비율 높을 때 상한 늘리기

## 2. 전체 shard 에 strict filter index — `build_str_index.py`

caption pass row index 만 shard 별 `.npy` 로 저장 (이미지 다운로드 X). resume 가능.

```bash
python build_str_index.py                       # 디폴트 8 workers, 4627 shards
python build_str_index.py --workers 16
python build_str_index.py --max_shards 10       # 작게 calibration
python build_str_index.py --force               # 기존 .npy 무시
```

실측 (10 shards, 8 workers): wall **60s**, rows **2.03M**, matched **709K**, **pass rate 34.86%** (`recap_str_index/SUMMARY.json`).

## 3. DBNet++ STD → crop — `crop_with_dbnetpp.py`

```bash
# 디폴트: ./recap_str_samples 에서 5장 처리, ./recap_str_crops 에 저장
python crop_with_dbnetpp.py

python crop_with_dbnetpp.py --n 50 --score_thr 0.3
python crop_with_dbnetpp.py --in_dir ./other_dir --out ./other_crops
```

산출물:
- `<out>/crops/<stem>__NN.jpg`
- `<out>_preview.html` (polygon overlay + crops)

전제: `data_env.sh` 로 mmocr 스택 설치 완료. weights 는 `~/STR/Union14M/checkpoints/dbnetpp_oclip.pth`.

---

## 환경

- `data_env.sh` — conda env `data` 에 HF stream + mmocr (DBNet++) 스택 설치
- 외부 의존: `~/STR/Union14M` (mmocr-dev-1.x config + `tools/rotate_crop.py`)

## 현재 산출물

| 폴더 | 내용 |
|---|---|
| `recap_datacomp_samples/` | filter 없이 받은 baseline 10장 |
| `recap_str_samples/` | `--filter str` 통과 100장 |
| `recap_str_crops/crops/` | DBNet++ 가 잘라낸 word-level crop (5장 → 68개) |
| `recap_str_index/` | 10 shards 의 caption-pass row index `.npy` + `SUMMARY.json` |
