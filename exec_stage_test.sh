#! /bin/bash

source .venv/bin/activate
nohup python3 -m processing_pipeline.s0_noise_filtering.runner_test > data/s0_test.txt 2>&1 &

