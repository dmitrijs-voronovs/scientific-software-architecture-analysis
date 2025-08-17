from cfg.LLMHost import LLMHost
from processing_pipeline.s2_arch_relevance_check.ArchRelevanceCheck_v2 import ArchitectureRelevanceCheckStage_v2


def main():
    ArchitectureRelevanceCheckStage_v2(hostname=LLMHost.SERVER, n_threads_override=1, batch_size_override=30).execute(
        [
            "google.deepvariant.v1.6.1.issue_comment"
        ], reverse=False)


if __name__ == "__main__":
    main()
