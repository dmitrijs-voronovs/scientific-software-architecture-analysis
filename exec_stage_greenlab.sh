#! /bin/bash

. .venv/bin/activate
nohup python3 -m processing_pipeline.s0_noise_filtering.NoiseFiltering > data/s0_greenlab.txt 2>&1 &

