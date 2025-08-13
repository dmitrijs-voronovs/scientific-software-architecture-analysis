from cfg.LLMHost import LLMHost
from processing_pipeline.s1_qa_relevance_check.QARelevanceCheck_v2 import QARelevanceCheckStage_v2


def main():
    QARelevanceCheckStage_v2(hostname=LLMHost.SERVER, cot_prompt=True, n_threads_override=10, batch_size_override=10).execute(["root-project.root.v6-32-06.code_comment.","root-project.root.v6-32-06.docs.","root-project.root.v6-32-06.issue."], reverse=True, dry_run=False)

    # QARelevanceCheckStage_v2().update_last_processed_item("root-project.root.v6-32-06.issue_comment.5.dat", 313)

    # cls = QARelevanceCheckStage_v2()
    # for fname in ["OpenGene.fastp.v0.23.4.issue_comment.parquet", "allenai.scispacy.v0.5.5.issue_comment.parquet",
    #               "root-project.root.v6-32-06.issue_comment.1.parquet", "scverse.scanpy.1.10.2.issue_comment.3.parquet",
    #               "scverse.scanpy.1.10.2.issue_comment.1.parquet", "google.deepvariant.v1.6.1.issue_comment.2.parquet",
    #               "root-project.root.v6-32-06.issue_comment.4.parquet",
    #               "root-project.root.v6-32-06.issue_comment.2.parquet",
    #               "root-project.root.v6-32-06.issue_comment.0.parquet",
    #               "google.deepvariant.v1.6.1.issue_comment.1.parquet", "scverse.scanpy.1.10.2.issue.parquet",
    #               "scverse.scanpy.1.10.2.issue_comment.0.parquet", "scverse.scanpy.1.10.2.issue_comment.2.parquet",
    #               "root-project.root.v6-32-06.issue_comment.3.parquet",
    #               "root-project.root.v6-32-06.issue_comment.6.parquet",
    #               "root-project.root.v6-32-06.issue_comment.5.parquet", ]:
    #     cls.clean_cache(fname)


if __name__ == "__main__":
    main()
