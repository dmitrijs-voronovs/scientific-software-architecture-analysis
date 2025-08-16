from cfg.LLMHost import LLMHost
from processing_pipeline.s2_arch_relevance_check.ArchRelevanceCheck_v2 import ArchitectureRelevanceCheckStage_v2


def main():
    ArchitectureRelevanceCheckStage_v2(hostname=LLMHost.SERVER).execute(
        ["root-project.root.v6-32-06.code_comment", "root-project.root.v6-32-06.docs",
         "google.deepvariant.v1.6.1.issue_comment"], reverse=True)


if __name__ == "__main__":
    main()
