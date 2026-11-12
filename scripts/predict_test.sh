#!/usr/bin/env bash
# 평가용 10장 추론 — 제출 명령 그대로. 설치(README 2절)를 마친 환경에서 이 폴더로 이동해 실행한다.
# 결과: runs/w2_res_s15+w2_res_s20/pred_test_nohalo_learned/*_mask.png
set -euo pipefail
cd "$(dirname "$0")"
python -m komap.predict \
  --exp w2_res_s15,w2_res_s20 \
  --prob4-exp w4_prob2_si2 \
  --combiner learned \
  --split test "$@"
