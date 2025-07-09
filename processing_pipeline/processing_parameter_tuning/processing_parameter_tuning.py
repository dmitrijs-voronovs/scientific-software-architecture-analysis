import itertools
import signal
import sys
import time

import pandas as pd

from cfg.LLMHost import LLMHost
from cfg.ModelName import ModelName
from constants.abs_paths import AbsDirPath
from processing_pipeline.s0_noise_filtering.NoiseFiltering import NoiseFilteringStage


def main():
    # warm up run
    # NoiseFilteringStage(hostname=LLMHost.GREEN_LAB, n_threads=10, batch_size=10, model_name_override=ModelName.DEEPSEEK_1_5B, disable_cache=True).execute(
    #     ["google.deepvariant.v1.6.1.code_comment"], reverse=False)

    # NoiseFilteringStage(hostname=LLMHost.RADU_SERVER, n_threads=10, batch_size=10, model_name_override=ModelName.DEEPSEEK_1_5B).execute_single_threaded(
    #     ["google.deepvariant.v1.6.1.code_comment"], reverse=False)

    n_threads = [1, 5, 10, 15]
    n_batches = [5, 10, 15, 20, 50]
    model_names = ModelName.all_models
    results = []
    for threads, batches, model_name in itertools.product(n_threads, n_batches, model_names):
        start = time.time()
        NoiseFilteringStage(hostname=LLMHost.GREEN_LAB, n_threads=threads, batch_size=batches,
                            model_name_override=model_name, disable_cache=True,
                            in_dir_override=AbsDirPath.PARAMETER_TUNING_DIR,
                            out_dir_override=AbsDirPath.PARAMETER_TUNING_RES_DIR).execute()
        end = time.time()
        results.append({"n_threads": threads, "n_batches": batches, "model_name": model_name, "time": end - start})
        print(f"Finished {model_name} with {threads} threads and {batches} batches in {end - start} seconds")
    df = pd.DataFrame(results)
    print(df)
    df.to_csv(AbsDirPath.RES_S0_NOISE_FILTERING / f"batch_thread_test_{LLMHost.GREEN_LAB}.csv", index=False)


if __name__ == "__main__":
    main()


def cleanup_and_exit(signal_num, frame):
    print("Caught interrupt, cleaning up...")
    sys.exit(0)  # Triggers the context manager's cleanup


signal.signal(signal.SIGINT, cleanup_and_exit)
