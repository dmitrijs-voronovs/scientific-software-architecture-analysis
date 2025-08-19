from cfg.LLMHost import LLMHost
from processing_pipeline.s2_arch_relevance_check.ArchRelevanceCheck_v2 import ArchitectureRelevanceCheckStage_v2


def main():
    ArchitectureRelevanceCheckStage_v2(hostname=LLMHost.SERVER, n_threads_override=6, batch_size_override=14).execute()


if __name__ == "__main__":
    main()
