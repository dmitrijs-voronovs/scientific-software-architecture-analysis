from cfg.LLMHost import LLMHost
from processing_pipeline.s3_tactic_extraction.TacticExtraction_v2 import TacticExtractionStage_v2


def main():
    TacticExtractionStage_v2(hostname=LLMHost.SERVER, n_threads_override=6, batch_size_override=12).execute([
        "allenai.scispacy.v0.5.5",
        "OpenGene.fastp.v0.23.4"
    ], reverse=False)


if __name__ == "__main__":
    main()
