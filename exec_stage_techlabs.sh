#! /bin/bash

source .venv/bin/activate
nohup python3 -m processing_pipeline.s0_noise_filtering.runner_2 > data/s0_techlabs.txt 2>&1 &

