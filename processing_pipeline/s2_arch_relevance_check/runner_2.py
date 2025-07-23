from cfg.LLMHost import LLMHost
from processing_pipeline.s2_arch_relevance_check.ArchRelevanceCheck import ArchitectureRelevanceCheckStage


def main():
    ArchitectureRelevanceCheckStage(hostname=LLMHost.SERVER).execute(["code_comment.", "issue."], reverse=True)


if __name__ == "__main__":
    main()
