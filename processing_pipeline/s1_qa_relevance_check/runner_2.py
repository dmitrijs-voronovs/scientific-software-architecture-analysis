from cfg.LLMHost import LLMHost
from processing_pipeline.s1_qa_relevance_check.QARelevanceCheck import QARelevanceCheckStage


def main():
    QARelevanceCheckStage(hostname=LLMHost.SERVER).execute(["root-project.root.v6-32-06.code_comment.","root-project.root.v6-32-06.docs.","root-project.root.v6-32-06.issue_comment."], reverse=True)


if __name__ == "__main__":
    main()
