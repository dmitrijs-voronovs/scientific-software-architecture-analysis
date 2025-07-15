from cfg.LLMHost import LLMHost
from cfg.ModelName import ModelName
from processing_pipeline.s0_noise_filtering.NoiseFiltering import NoiseFilteringStage


def main():
    NoiseFilteringStage(hostname=LLMHost.SERVER, model_name_override=ModelName.DEEPSEEK_1_5B).execute(["root-project"], reverse=True)


if __name__ == "__main__":
    main()
    