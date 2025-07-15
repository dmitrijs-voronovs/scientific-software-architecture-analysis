from cfg.LLMHost import LLMHost
from cfg.ModelName import ModelName
from processing_pipeline.s0_noise_filtering.NoiseFiltering import NoiseFilteringStage


def main():
    NoiseFilteringStage(hostname=LLMHost.SERVER, disable_cache=True, batch_size_override=5, n_threads_override=2, model_name_override=ModelName.DEEPSEEK_1_5B).execute(["root-project"], reverse=True)


if __name__ == "__main__":
    main()
