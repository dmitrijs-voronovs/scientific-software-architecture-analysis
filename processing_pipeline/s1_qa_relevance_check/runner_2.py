from cfg.LLMHost import LLMHost
from processing_pipeline.s1_qa_relevance_check.QARelevanceCheck import QARelevanceCheckStage
from processing_pipeline.s1_qa_relevance_check.QARelevanceCheck_v2 import QARelevanceCheckStage_v2


def main():
    # QARelevanceCheckStage_v2().update_last_processed_item("root-project.root.v6-32-06.issue_comment.5.dat", 313)
    QARelevanceCheckStage_v2(hostname=LLMHost.SERVER, cot_prompt=True, n_threads_override=5, batch_size_override=10).execute(["root-project.root.v6-32-06.code_comment.","root-project.root.v6-32-06.docs.","root-project.root.v6-32-06.issue."], reverse=True, dry_run=False)


if __name__ == "__main__":
    main()

