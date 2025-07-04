import math
import os
import re
import shelve
import signal
import sys
from pathlib import Path
from typing import List

import dotenv
import pandas as pd
from langchain_ollama import ChatOllama
from loguru import logger
from pydantic import BaseModel
from tenacity import RetryError
from tqdm import tqdm

from constants.abs_paths import AbsDirPath
from constants.foldernames import FolderNames
from cfg.selected_repos import selected_repos
from utilities.utils import create_logger_path


# Load environment variables from .env file
dotenv.load_dotenv()


class OllamaFormatValidityResponse(BaseModel):
    to_eliminate: bool
    reason: str

# @retry(stop=stop_after_attempt(6), wait=wait_fixed(3), after=lambda retry_state: logger.warning(retry_state),
#     reraise=True, )
def request_ollama_chain(prompts: List[str], base_url: str) -> List[OllamaFormatValidityResponse]:
    # model_name = "gemma"
    # model_name = "gemma2"
    model_name = "deepseek-r1:8b"
    model  = ChatOllama(model=model_name, temperature=0.0, base_url=base_url, format=OllamaFormatValidityResponse.model_json_schema())
    batch_answers = model.batch(prompts)
    return [OllamaFormatValidityResponse.model_validate_json(answer.content) for answer in batch_answers]

to_prompt = lambda x: \
f"""
You are an expert in analyzing and categorizing text content. Your task is to evaluate whether the given **target content** should be filtered out. The goal is to identify and **keep** content that consists of meaningful human-written prose, explanation, or analysis intended for human readers, and to **filter out** content that is primarily non-prose programmatic or technical artifacts intended mainly for machines or formal structure.

## Instructions:
For each input, return:
1. `to_eliminate`: true or false — should this content be eliminated?
2. `reasoning`: Brief explanation of why the decision was made.

### Keep Content That:
- Is written for human readers and contains **significant natural language, explanation, commentary, analysis, or discussion**.
- Reflects **communication intended for developers or users**, such as thoughtful suggestions, analysis, critiques, or explanations of implementation/optimization strategies.
- Includes **scientific, academic, or detailed technical discussions**, even if highly formal or specialized (e.g., detailed explanations of model architecture, reasoning behind design choices, analysis of outcomes).
- **Crucially:** This content should be kept **even if it is embedded within or formatted as** technical artifacts (like code comments, string literals in config files, documentation sections within code) **as long as the natural language prose component is substantial and provides meaningful human-readable context or explanation.**

### Eliminate Content That:
- Is **primarily** composed of non-prose programmatic or technical artifacts, **lacking significant natural language explanation or discussion**.
- Consists mainly of:
 - **Pure executable code or formal syntax** (e.g., function bodies without comments, simple variable declarations, pure boolean logic like `if (x > 5) {{ y = 1; }}` without explanation).
 - **Program output, logs, or error traces:** Content generated by programs (like build tools, compilers, runtime environments) for diagnostic or reporting purposes, characterized by structured formats, timestamps, error codes, etc., and **distinguished by the absence of substantial human-authored explanations or narrative.**
 - **Formal configuration, data structures, or build specifications lacking explanatory comments/text** (e.g., pure YAML/JSON data structures, simple Makefile rules, compiler flags lists without descriptive text).
 - **Version control metadata lacking explanatory commit messages** (e.g., diff hunks, merge conflict markers, simple file path changes without a descriptive commit message).
 - **Formal API signatures or technical interface definitions without accompanying prose** (e.g., `def my_function(param1: int) -> str:` without a docstring explaining *what* the function does or *why*).

## Examples (for reference only – do not analyze):

### Example 1
**Content:** Build failed on ROOT-ubuntu2004/python3.; Running on root-ubuntu-2004-3.cern.ch:/home/sftnight/build/...; Failing tests:; - projectroot.test.test_stressgraphics_interpreted
**Answer:**
to_eliminate: true
reasoning: Consists entirely of build logs and test failures, which are diagnostic artifacts, not human-readable prose explaining a concept.

### Example 2
**Content:** recision><conversion specifier>`` where:. * ``#`` is an optional flag available for hex values (see; ``<conversion specifier>`` below) which requires the value matched to be; prefixed by ``0x``.; * ``.<precision>`` is an optional printf-style precision specifier in which; ``<precision>`` indicates the minimum number of digits that the value matched; must have, expecting leading zeros if needed. * ``<conversion specifier>`` is an optional scanf-style conversion specifier; to indicate what number format to match (e.g. hex number). Currently; accepted format specifiers are ``%u``, ``%d``, ``%x`` and ``%X``.
**Answer:**
to_eliminate: true
reasoning: Primarily a formal technical specification of syntax with only minimal natural language labeling, not a substantial explanation.

### Example 3
**Content:** I tested the new parallelization strategy. Simulation time dropped 30%, but memory usage increased. We may need more efficient data structures.
**Answer:**
to_eliminate: false
reasoning: Natural language explanation of performance trade-offs.

### Example 4
**Content:** The MemoryDef structure now keeps two operands: the defining access and the optimized access. This change allows faster walking of Def chains and enables caching.
**Answer:**
to_eliminate: false
reasoning: Explains technical design changes in natural language with rationale.

### Example 5
**Content:** We propose SPECTER, a document-level embedding model trained using citation graphs. It improves scientific document classification without task-specific fine-tuning.
**Answer:**
to_eliminate: false
reasoning: Describes an academic NLP model in natural language.

### Example 6
**Content:** # Configure the learning rate using an exponential decay.
**Answer:**
to_eliminate: false
reasoning: Although formatted as a code comment, the content is natural language providing a meaningful explanation of a technical strategy and its purpose.

---

## Now analyze ONLY the following content:

**Content to evaluate:**
{x['sentence']}
"""


def cleanup_and_exit(signal_num, frame):
    print("Caught interrupt, cleaning up...")
    sys.exit(0)  # Triggers the context manager's cleanup

# Register the signal handler
signal.signal(signal.SIGINT, cleanup_and_exit)

MIN_WORD_COUNT = 10

def filter_df(df):
    df = df[df["quality_attribute"].isin(["Testability", "Energy Efficiency"])]
    df["word_count"] = df["sentence"].apply(lambda x: len(re.sub(r"[\W_]+", " ", x).strip().split()))
    df = df[df["word_count"] >= MIN_WORD_COUNT]
    return df


def verify_file_batched_llm(file_path: Path, res_filepath: Path, host: str, batch_size=10):
    os.makedirs(f".cache/{FolderNames.FORMAT_VALIDATION_DIR}/", exist_ok=True)
    with shelve.open(f".cache/{FolderNames.FORMAT_VALIDATION_DIR}/{file_path.stem}") as db:
        if db.get("processed", False):
            logger.info(f"File {file_path.stem} already processed")
            return
        logger.info(f"Processing {file_path.stem}")

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            logger.info(e)
            return

        # df = df[df["true_positive"] == True]
        # df = filter_df(df)

        last_idx = db.get("idx", 0)
        df = df.iloc[last_idx:].copy()
        if last_idx > 0:
            logger.info(f"Continuing from {last_idx}")
            res_filepath = res_filepath.with_suffix(f".from_{last_idx}.csv")

        df['format_prompt'] = df.apply(lambda x: to_prompt(x), axis=1)
        res = []

        for i in tqdm(range(0, len(df), batch_size), total=math.ceil(len(df) / batch_size), desc=f"Verifying {file_path.stem} in batches of {batch_size}"):
            batch_df = df.iloc[i:i + batch_size]
            prompts = batch_df['format_prompt'].tolist()

            res.extend(process_batch(batch_df, host, i, last_idx, prompts))

            df_to_save = df.iloc[:i + batch_size].copy()
            df_to_save['to_eliminate'], df_to_save['reason'] = zip(*res)
            df_to_save.to_csv(res_filepath, index=False)
            db["idx"] = last_idx + i + batch_size

        db['processed'] = True
        logger.info(f"Processed {file_path.stem}")


# TODO: RetryError does not exist anymore. Refactor. Think about the correct way to handle errors.
#  What should happen if batch fails? Should we continue with the next file or something else?
# When it fails it is likely not a problem with the file, but with the connection, thus no need to retry, just stop processing (after 3 more tries??)
def process_batch(batch_df, host, i, last_idx, prompts):
    try:
        responses = request_ollama_chain(prompts, host)  # New batch query
        processed_responses = [(r.to_eliminate, r.reason) for r in responses]
        return processed_responses
    except RetryError as error:
        logger.error(f"Retry error at batch starting index {last_idx + i}, {error}")
        responses = [(None, str(error))] * len(batch_df)
        return responses
    except Exception as e:
        logger.error(e)
        errors_for_termination = ["HTTPConnectionPool",
                                  "No connection could be made because the target machine actively refused it"]
        if any(error in str(e) for error in errors_for_termination):
            logger.error("HTTPConnectionPool error, exiting")
            exit(1)
        responses = [(None, str(e))] * len(batch_df)
        return responses


def validate_arch(host, only_files_containing_text: List[str] | None = None, reverse: bool = False):
    only_files_containing_text = only_files_containing_text or []
    optimized_keyword_folder = AbsDirPath.OPTIMIZED_KEYWORDS
    AbsDirPath.LOGS.mkdir(exist_ok=True)
    os.makedirs(AbsDirPath.S0_NOISE_FILTERING, exist_ok=True)
    logger.add(create_logger_path(AbsDirPath.S0_NOISE_FILTERING), mode="w")

    try:
        for file_path in optimized_keyword_folder.glob("*.csv"):
            if any(repo.dotted_ref in file_path.stem for repo in selected_repos):
                keep_processing = len(only_files_containing_text) == 0 or any(text_to_test in file_path.stem for text_to_test in only_files_containing_text)
                if keep_processing == reverse:
                    continue

                res_filepath = AbsDirPath.S0_NOISE_FILTERING / f"{file_path.stem}.arch_verified.csv"
                verify_file_batched_llm(file_path, res_filepath, host,
                                        10)  # res_filepath = file_path.with_stem("test123")
    except Exception as e:
        logger.error(e)
        raise e


LOCAL_LLM_HOST = "http://localhost:11434"

if __name__ == "__main__":
    validate_arch(LOCAL_LLM_HOST, [
        "root-project"
    ], reverse=True)
