#! /bin/bash

. .venv/bin/activate
#nohup python3 -m processing_pipeline.s1_qa_relevance_check.QARelevanceCheck > data/processing_logs/s1_greenlab.txt 2>&1 &
#nohup python3 -m processing_pipeline.s1_qa_relevance_check.runner_2 > data/processing_logs/s1_techlab.txt 2>&1 &

#nohup python3 -m processing_pipeline.s2_arch_relevance_check.ArchRelevanceCheck > data/processing_logs/s2_greenlab.txt 2>&1 &
#nohup python3 -m processing_pipeline.s2_arch_relevance_check.runner_2 > data/processing_logs/s2_techlab.txt 2>&1 &

nohup python3 -m processing_pipeline.s3_tactic_extraction.TacticExtraction > data/processing_logs/s3_greenlab.txt 2>&1 &
nohup python3 -m processing_pipeline.s3_tactic_extraction.runner_2 > data/processing_logs/s3_techlab.txt 2>&1 &



