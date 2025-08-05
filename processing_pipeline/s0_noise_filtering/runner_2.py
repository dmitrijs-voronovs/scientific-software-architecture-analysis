from cfg.LLMHost import LLMHost
from processing_pipeline.s0_noise_filtering.NoiseFiltering import NoiseFilteringStage


def main():
    # NoiseFilteringStage(hostname=LLMHost.SERVER).execute(["root-project"], reverse=True)
    NoiseFilteringStage(hostname=LLMHost.TECH_LAB, disable_cache=True, batch_size_override=10,
                        n_threads_override=5).execute(["docs"], reverse=True)

if __name__ == "__main__":
    main()
    