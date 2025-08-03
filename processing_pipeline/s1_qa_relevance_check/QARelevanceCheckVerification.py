from cfg.LLMHost import LLMHost
from processing_pipeline.model.IStageVerification import IStageVerification
from processing_pipeline.s1_qa_relevance_check.QARelevanceCheck import QARelevanceCheckStage


class QARelevanceCheckVerification(IStageVerification):
    stage_to_verify = QARelevanceCheckStage

    source_columns = ['qa', "sentence"]
    ai_output_columns = ['true_positive', 'reasoning']


def main():
    QARelevanceCheckVerification(hostname=LLMHost.TECH_LAB, batch_size_override=20).execute_verification()


if __name__ == "__main__":
    main()
