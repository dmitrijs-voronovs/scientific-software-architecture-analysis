import itertools
import time

import pandas as pd

from cfg.LLMHost import LLMHost
from constants.abs_paths import AbsDirPath
from processing_pipeline.s0_noise_filtering.NoiseFiltering import NoiseFilteringStage

def main():
    # warm up run
    NoiseFilteringStage(hostname=LLMHost.GREEN_LAB, n_threads=10, batch_size=10).execute(["google.deepvariant.v1.6.1.CODE_COMMENT"], reverse=False)

    n_threads = [1, 5, 10, 15, 20]
    n_batches = [1, 5, 10, 15, 20, 50]
    results = []
    for threads, batches in itertools.product(n_threads, n_batches):
        start = time.time()
        NoiseFilteringStage(hostname=LLMHost.GREEN_LAB, n_threads=threads, batch_size=batches).execute(["google.deepvariant"], reverse=False)
        end = time.time()
        results.append({"n_threads": threads, "n_batches": batches, "time": end - start})
    df = pd.DataFrame(results)
    print(df)
    df.to_csv(AbsDirPath.RES_S0_NOISE_FILTERING / "batch_thread_test.csv", index=False)



if __name__ == "__main__":
    main()
