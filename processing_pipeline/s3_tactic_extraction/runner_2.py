from cfg.LLMHost import LLMHost
from processing_pipeline.s3_tactic_extraction.TacticExtraction import TacticExtractionStage


def main():
    TacticExtractionStage(hostname=LLMHost.SERVER).execute(["issue", "issue_comment"], reverse=False)


if __name__ == "__main__":
    main()
