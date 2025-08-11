#! /bin/bash

. .venv/bin/activate
nohup python3 -m processing_pipeline.s1_qa_relevance_check.QARelevanceCheck_v2 > data/processing_logs/v2_s1_greenlab.txt 2>&1 &
nohup python3 -m processing_pipeline.s1_qa_relevance_check.runner_2 > data/processing_logs/v2_s1_techlab.txt 2>&1 &

nohup python3 -m processing_pipeline.s2_arch_relevance_check.ArchRelevanceCheck_v2 > data/processing_logs/v2_s2_greenlab.txt 2>&1 &
nohup python3 -m processing_pipeline.s2_arch_relevance_check.runner_2 > data/processing_logs/v2_s2_techlab.txt 2>&1 &

nohup python3 -m processing_pipeline.s3_tactic_extraction.TacticExtraction_v2 > data/processing_logs/v2_s3_greenlab.txt 2>&1 &
nohup python3 -m processing_pipeline.s3_tactic_extraction.runner_2 > data/processing_logs/v2_s3_techlab.txt 2>&1 &



