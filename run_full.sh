#!/bin/bash
# Full run: README command for OPT-1.3B (128 calibration samples). Extra arguments are forwarded.
set -e
python3 run.py facebook/opt-1.3b c4 braq --blocksize 128 --salient_metric hessian "$@"
