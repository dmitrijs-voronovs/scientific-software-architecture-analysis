import json
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
import requests
from langchain_ollama import ChatOllama
from loguru import logger
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, RetryError, wait_fixed
from tqdm import tqdm

from constants.foldernames import FolderNames
from extract_quality_attribs_from_docs import MatchSource
from metadata.repo_info.repo_info import credential_list
from utils.utils import create_logger_path

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
    model  = ChatOllama(model=model_name, base_url=base_url, format=OllamaFormatValidityResponse.model_json_schema())
    batch_answers = model.batch(prompts)
    return [OllamaFormatValidityResponse.model_validate_json(answer.content) for answer in batch_answers]


to_prompt = lambda x: \
f"""You are an expert in analyzing and categorizing text content. Your task is to evaluate whether the given text contains meaningful human-readable sentences or if it consists primarily of logs, code samples, or programmatic API description that should be filtered out.

For each input text, analyze it and determine:
1. Whether it should be eliminated (true/false)
2. The reason for elimination (if applicable)

Evaluation criteria:
- Eliminate text that consists primarily of:
  * Code snippets or samples (marked by syntax, keywords like "if/else", brackets, etc.)
  * Program logs or error messages (timestamps, error codes, stack traces)
  * API documentation or specifications (parameter lists, return types)
  * Configuration files or build system output
  * Version control metadata or comments
  * Compiler/interpreter output or warnings
- Keep text that contains:
  * Complete, meaningful sentences in natural language
  * Explanatory or descriptive content
  * Human-written prose discussing concepts or ideas

Content: {x['sentence']}
"""

def cleanup_and_exit(signal_num, frame):
    print("Caught interrupt, cleaning up...")
    sys.exit(0)  # Triggers the context manager's cleanup

# Register the signal handler
signal.signal(signal.SIGINT, cleanup_and_exit)


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

            try:
                responses = request_ollama_chain(prompts, host)  # New batch query
                processed_responses = [(r.to_eliminate, r.reason) for r in responses]
                res.extend(processed_responses)
            except RetryError as error:
                logger.error(f"Retry error at batch starting index {last_idx + i}, {error}")
                responses = [(None, str(error))] * len(batch_df)
                res.extend(responses)
            except Exception as e:
                logger.error(e)
                errors_for_termination = ["HTTPConnectionPool",
                                          "No connection could be made because the target machine actively refused it"]
                if any(error in str(e) for error in errors_for_termination):
                    logger.error("HTTPConnectionPool error, exiting")
                    exit(1)
                responses = [(None, str(e))] * len(batch_df)
                res.extend(responses)

            df_to_save = df.iloc[:i + batch_size].copy()
            df_to_save['to_eliminate'], df_to_save['reason'] = zip(*res)
            df_to_save.to_csv(res_filepath, index=False)
            db["idx"] = last_idx + i + batch_size

        db['processed'] = True
        logger.info(f"Processed {file_path.stem}")


def validate_arch(host, only_files_containing_text: List[str] | None = None, reverse: bool = False):
    only_files_containing_text = only_files_containing_text or []
    keyword_folder = Path("metadata/keywords/")
    optimized_keyword_folder = keyword_folder / FolderNames.OPTIMIZED_KEYWORD_DIR
    os.makedirs(".logs", exist_ok=True)
    os.makedirs(keyword_folder / FolderNames.FORMAT_VALIDATION_DIR, exist_ok=True)
    logger.add(create_logger_path(FolderNames.FORMAT_VALIDATION_DIR), mode="w")

    try:
        for file_path in optimized_keyword_folder.glob("*.csv"):
            if any(cred.get_ref(".") in file_path.stem for cred in (credential_list)):
                keep_processing = len(only_files_containing_text) == 0 or any(text_to_test in file_path.stem for text_to_test in only_files_containing_text)
                if keep_processing == reverse:
                    continue

                res_filepath = keyword_folder / f"{FolderNames.FORMAT_VALIDATION_DIR}/{file_path.stem}.arch_verified.csv"
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
