#!/bin/bash
cd /home/nvidia/tac_r2s2r/experiments_v2
PY=~/isaacsim-venv/bin/python
$PY tacpp.py --stage main --seeds 11 23 37 53 71 83 97 109 127 149 --out results > log_main.txt 2>&1
$PY tacpp.py --stage sweep --seeds 11 23 37 53 71 --out results > log_sweep.txt 2>&1
$PY tacpp.py --stage diag --seeds 11 23 37 --out results > log_diag.txt 2>&1
echo ALLDONE
