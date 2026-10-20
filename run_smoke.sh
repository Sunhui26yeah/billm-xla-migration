#!/bin/bash
# Smoke test: BiLLM on OPT-1.3B with a capped calibration set (32 instead of 128 samples).
# Phase 1 quantizes all layers, saves the model and evaluates PPL; phase 2 reloads the saved model and re-evaluates.
# Extra arguments (e.g. --device) are forwarded to both phases.
set -e
python3 run.py facebook/opt-1.3b c4 braq --blocksize 128 --salient_metric hessian --nsamples 32 --save "$@"
python3 run.py facebook/opt-1.3b c4 braq --blocksize 128 --salient_metric hessian --nsamples 32 --load_quantized "$@"
