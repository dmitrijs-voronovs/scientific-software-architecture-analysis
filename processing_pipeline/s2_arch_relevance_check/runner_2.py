from cfg.LLMHost import LLMHost
from processing_pipeline.s2_arch_relevance_check.ArchRelevanceCheck_v2 import ArchitectureRelevanceCheckStage_v2


def main():
    ArchitectureRelevanceCheckStage_v2(hostname=LLMHost.SERVER, n_threads_override=10, batch_size_override=10).execute(
        ["root-project.root.v6-32-06.docs", "google.deepvariant.v1.6.1.issue_comment",
         "root-project.root.v6-32-06.code_comment.0.part_0.parquet",
         "root-project.root.v6-32-06.code_comment.0.part_1.parquet",
         "root-project.root.v6-32-06.code_comment.0.part_2.parquet",
         "root-project.root.v6-32-06.code_comment.0.part_3.parquet",
         "root-project.root.v6-32-06.code_comment.1.part_0.parquet",
         "root-project.root.v6-32-06.code_comment.1.part_1.parquet",
         "root-project.root.v6-32-06.code_comment.1.part_2.parquet"], reverse=True)


if __name__ == "__main__":
    main()
