import math
import os
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

from constants.foldernames import FolderNames
from cfg.repo_credentials import selected_credentials
from utils.utils import create_logger_path

# Load environment variables from .env file
dotenv.load_dotenv()


false_positive_prompt = """
You are an expert in evaluating and categorizing quality attributes in software engineering. You possess the necessary skills to distinguish sentences that clearly relate to a given quality attribute from those that do not. 

Evaluate whether the content accurately aligns with its associated quality attribute, given the context provided. Your goal is to determine if the content makes sense in relation to the quality attribute description or if it is a false positive.

Data:

Quality Attribute: Usability
Attribute Description: The degree to which users can effectively and efficiently accomplish tasks, including support for error recovery and user satisfaction. Usability covers ease of learning, efficient usage, and adaptability to user needs.
Content: Just learned `awk '$5=""*""'` replaces `T,*` with `*` and the correct usage is `awk '5~""*""'`.

Instructions: 
1. Analyze the content and the attribute description.
2. Determine if the content accurately reflects the intended quality attribute in this context.
3. If it does, label it as `true_positive: true`; if not, mark it as a `true_positive: false`.
4. If the content consists only of logs with no other text, mark it as a `true_positive: false`. 
4. If the content partially consists of logs, focus on analyzing remaining text. 
5. Add `reasoning` why the content is a true or false positive.
"""

true_positive_prompt = """
You are an expert in evaluating and categorizing quality attributes in software engineering. You possess the necessary skills to distinguish sentences that clearly relate to a given quality attribute from those that do not. 

Evaluate whether the content accurately aligns with its associated quality attribute, given the context provided. Your goal is to determine if the content makes sense in relation to the quality attribute description or if it is a false positive.

Data:

Quality Attribute: Deployability
Attribute Description: The capability of software to be deployed into an operational environment with predictable time and effort, including options for rollback if needed. Key aspects include automation, deployment speed, and deployment granularity.
Content: ies are installed from the correct channel and compiled against MKL; - conda-forge; - defaults; dependencies:. # core python dependencies; - conda-forge::python=3.6.10 # do not update; - pip=20.0.2 # specifying channel may cause a warning to be emitted by conda; - conda-forge::mkl=2019.5 # MKL typically provides dramatic performance increases for theano, tensorflow, and other key dependencies; - conda-forge::mkl-service=2.3.0; - conda-forge::numpy=1.17.5 # do not update, this will break scipy=0.19.1; # verify that numpy is compiled against MKL (e.g., by checking *_mkl_info using numpy.show_config()); # and that it is used in tensorflow, theano, and other key dependencies; - conda-forge::theano=1.0.4 # it is unlikely that new versions of theano will be released; # verify that this is using numpy compiled against MKL (e.g., by the presence of -lmkl_rt in theano.config.blas.ldflags); - defaults::tensorflow=1.15.0 # update only if absolutely necessary, as this may cause conflicts with other core dependencies; # verify that this is using numpy compiled against MKL (e.g., by checking tensorflow.pywrap_tensorflow.IsMklEnabled()); - conda-forge::scipy=1.0.0 # do not update, this will break a scipy.misc.logsumexp import (deprecated in scipy=1.0.0) in pymc3=3.1; - conda-forge::pymc3=3.1 # do not update, this will break gcnvkernel; - conda-forge::keras=2.2.4 # updated from pip-installed 2.2.0, which caused various conflicts/clobbers of conda-installed packages; # conda-installed 2.2.4 appears to be the most recent version with a consistent API and without conflicts/clobbers; # if you wish to update, note that versions of conda-forge::keras after 2.2.5; # undesirably set the environment variable KERAS_BACKEND = theano by default; - defaults::intel-openmp=2019.4; - conda-forge::scikit-learn=0.22.2; - conda-forge::matplotlib=3.2.1; - conda-forge::pandas=1.0.3. # core R dependencies; these should only be used for plotting and do not take precedence over core python dependencies!; -

Instructions: 
1. Analyze the content and the attribute description.
2. Determine if the content accurately reflects the intended quality attribute in this context.
3. If it does, label it as `true_positive: true`; if not, mark it as a `true_positive: false`.
4. If the content consists only of logs with no other text, mark it as a `true_positive: false`. 
4. If the content partially consists of logs, focus on analyzing remaining text. 
5. Add `reasoning` why the content is a true or false positive.
"""


class OllamaArchitectureResponse(BaseModel):
    related_to_architecture: bool
    related_to_architecture_reasoning: str

# @retry(stop=stop_after_attempt(6), wait=wait_fixed(3), after=lambda retry_state: logger.warning(retry_state),
#     reraise=True, )
def request_ollama_chain(prompts: List[str], base_url: str) -> List[OllamaArchitectureResponse]:
    # model_name = "gemma"
    # model_name = "gemma2"
    model_name = "deepseek-r1:8b"
    model  = ChatOllama(model=model_name, base_url=base_url, format=OllamaArchitectureResponse.model_json_schema())
    batch_answers = model.batch(prompts)
    return [OllamaArchitectureResponse.model_validate_json(answer.content) for answer in batch_answers]


to_prompt = lambda x: f"""
You are an expert in evaluating and categorizing quality attributes in software engineering. You possess the necessary skills to distinguish sentences that clearly relate to a given quality attribute from those that do not. 

Evaluate whether the content accurately aligns with its associated quality attribute, given the context provided. Your goal is to determine if the content makes sense in relation to the quality attribute description or if it is a false positive.

Data:

Quality Attribute: {x['quality_attribute']}
Attribute Description: {x['attribute_desc']}
Content: {x['sentence']}

Instructions: 
1. Analyze the content and the attribute description.
2. Determine if the content accurately reflects the intended quality attribute in this context.
3. If it does, label it as an accurate match; if not, mark it as a false positive.
4. Output only the JSON object in response, without any additional explanation.
5. Ensure the JSON output is properly formatted. Escape any special characters or inner quotes in strings to ensure compatibility with JSON parsers. Within JSON strings use \\\" to escape double quotes.

Output your response as a JSON object in the following format:
{{
  "false_positive": <boolean>,
  "reasoning": "<str>"
}}
"""

to_prompt_about_architecture = lambda x: f"""
You are an expert in software architecture and software engineering. You have the necessary expertise to evaluate whether a given piece of content is related to software architecture.

Evaluate whether the content explicitly discusses or relates to software architecture concepts, principles, or concerns. Your goal is to determine if the content is relevant to software architecture.

Data:

Content: {x['sentence']}
Instructions:

1. Analyze the content and determine whether it is discussing software architecture, including but not limited to:
    * Architectural patterns or styles (e.g., microservices, monolith, event-driven architecture).
    * Architectural decisions, trade-offs, or quality attributes (e.g., scalability, maintainability, performance).
    * High-level system structure, interactions, dependencies, or constraints.
2. If the content clearly pertains to software architecture, mark it as `related_to_architecture: true`.
3. If the content is general software development, code-level details, logs, or unrelated to architecture, mark it as `related_to_architecture: false`.
4. If the content includes partial architectural relevance but is mostly about implementation details, analyze whether the relevant part is strong enough to classify it as `related_to_architecture: true`.
5. Provide `related_to_architecture_reasoning` explaining why the content is classified as related on unrelated.
"""

def cleanup_and_exit(signal_num, frame):
    print("Caught interrupt, cleaning up...")
    sys.exit(0)  # Triggers the context manager's cleanup

# Register the signal handler
signal.signal(signal.SIGINT, cleanup_and_exit)


def verify_file_batched_llm(file_path: Path, res_filepath: Path, host: str, batch_size=10):
    os.makedirs(f".cache/{FolderNames.ARCHITECTURE_VERIFICATION_DIR}/", exist_ok=True)
    with shelve.open(f".cache/{FolderNames.ARCHITECTURE_VERIFICATION_DIR}/{file_path.stem}") as db:
        if db.get("processed", False):
            logger.info(f"File {file_path.stem} already processed")
            return
        logger.info(f"Processing {file_path.stem}")

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            logger.info(e)
            return

        df = df[df["true_positive"] == True]
        last_idx = db.get("idx", 0)
        df = df.iloc[last_idx:].copy()
        if last_idx > 0:
            logger.info(f"Continuing from {last_idx}")
            res_filepath = res_filepath.with_suffix(f".from_{last_idx}.csv")

        df['arch_prompt'] = df.apply(lambda x: to_prompt_about_architecture(x), axis=1)
        res = []

        for i in tqdm(range(0, len(df), batch_size), total=math.ceil(len(df) / batch_size), desc=f"Verifying {file_path.stem} in batches of {batch_size}"):
            batch_df = df.iloc[i:i + batch_size]
            prompts = batch_df['arch_prompt'].tolist()

            try:
                responses = request_ollama_chain(prompts, host)  # New batch query
                processed_responses = [(r.related_to_architecture, r.related_to_architecture_reasoning) for r in responses]
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
            df_to_save['related_to_architecture'], df_to_save['related_to_architecture_reasoning'] = zip(*res)
            df_to_save.to_csv(res_filepath, index=False)
            db["idx"] = last_idx + i + batch_size

        db['processed'] = True
        logger.info(f"Processed {file_path.stem}")


def validate_arch(host, only_files_containing_text: List[str] | None = None, reverse: bool = False):
    only_files_containing_text = only_files_containing_text or []
    keyword_folder = Path("metadata/keywords/")
    optimized_keyword_folder = keyword_folder / FolderNames.VERIFICATION_DIR
    os.makedirs("../../.logs", exist_ok=True)
    os.makedirs(keyword_folder / FolderNames.ARCHITECTURE_VERIFICATION_DIR, exist_ok=True)
    logger.add(create_logger_path(FolderNames.ARCHITECTURE_VERIFICATION_DIR), mode="w")

    try:
        for file_path in optimized_keyword_folder.glob("*.csv"):
            if any(cred.get_ref(".") in file_path.stem for cred in (selected_credentials)):
                keep_processing = len(only_files_containing_text) == 0 or any(text_to_test in file_path.stem for text_to_test in only_files_containing_text)
                if keep_processing == reverse:
                    continue

                res_filepath = keyword_folder / f"{FolderNames.ARCHITECTURE_VERIFICATION_DIR}/{file_path.stem}.arch_verified.csv"
                verify_file_batched_llm(file_path, res_filepath, host,
                                        10)  # res_filepath = file_path.with_stem("test123")
    except Exception as e:
        logger.error(e)
        raise e


LOCAL_LLM_HOST = "http://localhost:11435"

if __name__ == "__main__":
    validate_arch(LOCAL_LLM_HOST, [
        "root-project.root.v6-32-06.WIKI.1",
        "root-project.root.v6-32-06.WIKI.2",
        "root-project.root.v6-32-06.WIKI.3",
        "root-project.root.v6-32-06.WIKI.4",
        "root-project.root.v6-32-06.WIKI.5",
        "root-project.root.v6-32-06.WIKI.6",
    ], False)
