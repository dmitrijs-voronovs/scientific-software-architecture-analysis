from cfg.LLMHost import LLMHost
from processing_pipeline.s0_noise_filtering.NoiseFiltering import NoiseFilteringStage


def main():
    NoiseFilteringStage(hostname=LLMHost.SERVER).execute(["root-project"], reverse=True)


if __name__ == "__main__":
    main()
    