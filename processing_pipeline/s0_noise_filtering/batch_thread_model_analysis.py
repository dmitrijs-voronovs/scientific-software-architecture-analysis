import itertools
import time

import pandas as pd

from cfg.LLMHost import LLMHost
from cfg.ModelName import ModelName
from constants.abs_paths import AbsDirPath
from processing_pipeline.s0_noise_filtering.NoiseFiltering import NoiseFilteringStage


def main():
    # warm up run
    NoiseFilteringStage(hostname=LLMHost.RADU_SERVER, n_threads=1, batch_size=1, model_name_override=ModelName.DEEPSEEK_1_5B).execute(
        ["google.deepvariant.v1.6.1.code_comment"], reverse=False)

    n_threads = [1, 5, 10, 15, 20]
    n_batches = [1, 5, 10, 15, 20, 50]
    model_names = ModelName.all_models
    results = []
    for threads, batches, model_name in itertools.product(n_threads, n_batches, model_names):
        start = time.time()
        NoiseFilteringStage(hostname=LLMHost.RADU_SERVER, n_threads=threads, batch_size=batches).execute(
            ["google.deepvariant"], reverse=False)
        end = time.time()
        results.append({"n_threads": threads, "n_batches": batches, "model_name": model_name, "time": end - start})
    df = pd.DataFrame(results)
    print(df)
    df.to_csv(AbsDirPath.RES_S0_NOISE_FILTERING / "batch_thread_test.csv", index=False)


if __name__ == "__main__":
    main()
