#! /bin/bash

. .venv/bin/activate
nohup python3 -m processing_pipeline.s0_noise_filtering.NoiseFiltering > data/s1_greenlab.txt 2>&1 &
nohup python3 -m processing_pipeline.s0_noise_filtering.NoiseFiltering > data/ws1_techlab.txt 2>&1 &

