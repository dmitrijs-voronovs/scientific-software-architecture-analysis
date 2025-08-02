from cfg.LLMHost import LLMHost
from processing_pipeline.model.IStageVerification import IStageVerification
from processing_pipeline.s2_arch_relevance_check.ArchRelevanceCheck import ArchitectureRelevanceCheckStage


class ArchitectureRelevanceCheckVerification(IStageVerification):
    stage_to_verify = ArchitectureRelevanceCheckStage

    source_columns = ['sentence']
    ai_output_columns = ['related_to_arch', 'reasoning']


def main():
    ArchitectureRelevanceCheckVerification(hostname=LLMHost.RADU_SERVER, batch_size_override=20).execute_verification()


if __name__ == "__main__":
    main()
