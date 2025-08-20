from cfg.LLMHost import LLMHost
from processing_pipeline.s3_tactic_extraction.TacticExtraction_v2 import TacticExtractionStage_v2


def main():
    TacticExtractionStage_v2(hostname=LLMHost.SERVER, n_threads_override=4, batch_size_override=8).execute(
        ["issue.", "code_comment."], reverse=False)

if __name__ == "__main__":
    main()
