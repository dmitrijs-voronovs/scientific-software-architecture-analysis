from cfg.LLMHost import LLMHost
from processing_pipeline.model.IStageVerification import IStageVerification
from processing_pipeline.s3_tactic_extraction.TacticExtraction import TacticExtractionStage


class TacticExtractionVerification(IStageVerification):
    stage_to_verify = TacticExtractionStage

    source_columns = ['qa', 'sentence']
    ai_output_columns = ['tactic', 'response']


def main():
    TacticExtractionVerification(hostname=LLMHost.TECH_LAB, batch_size_override=20).execute_verification()


if __name__ == "__main__":
    main()
