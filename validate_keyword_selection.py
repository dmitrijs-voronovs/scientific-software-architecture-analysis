import json
import os
import re
import shelve
import signal
import sys
from datetime import datetime
from pathlib import Path

import dotenv
import pandas as pd
import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, RetryError, wait_incrementing, wait_fixed
from tqdm import tqdm

from extract_quality_attribs_from_docs import MatchSource
from metadata.repo_info.repo_info import credential_list
from model.Credentials import Credentials
from utils.utils import create_logger_path

# Load environment variables from .env file
dotenv.load_dotenv()

# OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_TOKEN_2")

YOUR_APP_NAME = "Keyword Selection"


def request_llm(prompt):
    response = requests.post(url="https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "X-Title": f"{YOUR_APP_NAME}",
            # Optional. Shows in rankings on openrouter.ai.
        }, data=json.dumps({"model": "google/gemma-2-9b-it:free",  # Optional
            "messages": [{"role": "system",
                "content": "You are a helpful assistant who provides thoughtful and concise responses and tightly follows user instroctuctions. You never output any extra information."},
                {"role": "user", "content": prompt}]

        }))
    return response.json()


def get_resp(index, row):
    try:
        resp = request_llm(row["prompt"])
        if resp['error']:
            print(f"Error in row {index}, {resp['error']}")
            exit(1)
        text_resp = resp['choices'][0]['message']['content']
        text_resp = re.sub(r'```json|```', "", text_resp)
        json_resp = json.loads(text_resp)
        print(json_resp)
        r = (json_resp['false_positive'], json_resp['reasoning'])
    except Exception as e:
        print(f"Error in row {index}, {e}")
        r = (None, None)
    return r


@retry(stop=stop_after_attempt(8), wait=wait_incrementing(10, 10),
    before_sleep=lambda retry_state: print(retry_state), )
def request_google_ailab(model, prompt):
    response = model.generate_content(prompt)
    text_resp = re.sub(r'```json|```', "", response.text)
    json_resp = json.loads(text_resp)
    print(json_resp)
    return json_resp['false_positive'], json_resp['reasoning']


LOCAL_LLM_HOST = "http://localhost:11434"


@retry(stop=stop_after_attempt(6), wait=wait_fixed(3), after=lambda retry_state: logger.warning(retry_state),
    reraise=True, )
def request_gemma(prompt):
    url = "%s/api/generate" % LOCAL_LLM_HOST

    payload = json.dumps({"model": "gemma", "prompt": prompt, "stream": False})
    response = requests.request("POST", url, headers={'Content-Type': 'application/json'}, data=payload).json()

    try:
        text_resp = re.sub(r'```json|```', "", response['response'])
        json_resp = json.loads(text_resp)
        return json_resp['false_positive'], json_resp['reasoning']
    except Exception as e:
        raise Exception(f"Error in response: {response['response']}", e)


quality_attribs = {"Availability": {
    "desc": "The system's readiness to perform its function when required, focusing on reliability and recovery. It involves fault masking or repair to prevent failures, ensuring minimal cumulative downtime."},
    "Deployability": {
        "desc": "The capability of software to be deployed into an operational environment with predictable time and effort, including options for rollback if needed. Key aspects include automation, deployment speed, and deployment granularity."},
    "Energy Efficiency": {
        "desc": "The system’s ability to optimize resource use and minimize energy consumption while achieving required performance. This involves monitoring, allocation, and adaptation of resources."},
    "Integrability": {
        "desc": "The ease of combining the system with other systems or components, measured by integration cost and technical risks. Integrability considers the complexity and compatibility of interfaces, including syntactic, semantic, behavioral, and temporal alignment."},
    "Modifiability": {
        "desc": "The ease with which the system can be adapted by adding, removing, or modifying features, or adjusting to new environments. This attribute involves assessing the time, cost, and impact of changes, considering factors like coupling, cohesion, and the scope of modifications."},
    "Performance": {
        "desc": "The system’s capacity to meet its timing requirements, managing event handling and response times effectively. Performance focuses on reducing blocked time from resource contention and optimizing resource utilization under varying load conditions."},
    "Safety": {
        "desc": "The system’s ability to avoid states that could lead to harm or damage. Safety encompasses detection and handling of errors (e.g., omissions, timing, incorrect values) to prevent hazardous outcomes or mitigate potential damage."},
    "Security": {
        "desc": "The system’s ability to safeguard information against unauthorized access, while permitting authorized access. Security emphasizes confidentiality, integrity, and availability, using tactics to detect, prevent, and respond to attacks."},
    "Testability": {
        "desc": "The ease of validating software functionality through testing, enabling fault detection. This includes controlling and observing the system’s state, reducing complexity, and facilitating the creation of test cases and oracles."},
    "Usability": {
        "desc": "The degree to which users can effectively and efficiently accomplish tasks, including support for error recovery and user satisfaction. Usability covers ease of learning, efficient usage, and adaptability to user needs."}}

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

def cleanup_and_exit(signal_num, frame):
    print("Caught interrupt, cleaning up...")
    sys.exit(0)  # Triggers the context manager's cleanup

# Register the signal handler
signal.signal(signal.SIGINT, cleanup_and_exit)

verification_dir = "verification_v2"

def verify_file(file_path: Path, res_filepath: Path, batch_size=10):
    os.makedirs(f".cache/{verification_dir}/", exist_ok=True)
    with shelve.open(f".cache/{verification_dir}/{file_path.stem}") as db:
        if db.get("processed", False):
            logger.info(f"File {file_path.stem} already processed")
            return
        logger.info(f"Processing {file_path.stem}")

        # genai.configure(api_key=os.getenv("GOOGLE_AI_STUDIO_KEY"))
        # model = genai.GenerativeModel("gemini-1.5-flash")

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            logger.info(e)
            return

        df = df.groupby(["quality_attribute", "sentence", "keyword"]).first().reset_index()

        last_idx = db.get("idx", 0)
        df = df.iloc[last_idx:].copy()
        if last_idx > 0:
            logger.info(f"Continuing from {last_idx}")
            res_filepath = res_filepath.with_suffix(f".from_{last_idx}.csv")

        df["attribute_desc"] = df["quality_attribute"].apply(lambda x: quality_attribs[x]["desc"])
        df['prompt'] = df.apply(lambda x: to_prompt(x), axis=1)
        res = []
        for i, (_idx, row) in tqdm(enumerate(df.iterrows()), total=df.shape[0], desc=f"Verifying {file_path.stem}"):
            try:
                # r = request_google_ailab(model, row["prompt"])
                try:
                    r = request_gemma(row["prompt"])
                    res.append(r)
                except RetryError as error:
                    logger.error(f"Retry error, current_element={last_idx + i + 1}, {row=}, {error}")
                except Exception as e:
                    logger.error(e)
                    if "HTTPConnectionPool" in str(e):
                        logger.error("HTTPConnectionPool error, exiting")
                        exit(1)
                    res.append((None, str(e)))

                if (i + 1) % batch_size == 0:
                    df_to_save = df.iloc[0:i + 1].copy()
                    df_to_save['false_positive'], df_to_save['reasoning'] = zip(*res)
                    df_to_save.to_csv(res_filepath, index=False)
                    db["idx"] = last_idx + i + 1

            except Exception as e:
                print(f"Error in row {i + 1}, {e}")
        df['false_positive'], df['reasoning'] = zip(*res)
        df.to_csv(res_filepath, index=False)

        db['processed'] = True
        logger.info(f"Processed {file_path.stem}")


OPTIMIZED_KEYWORD_FOLDER_NAME = "optimized"

def main():
    keyword_folder = Path("metadata/keywords/")
    optimized_keyword_folder = keyword_folder / OPTIMIZED_KEYWORD_FOLDER_NAME
    os.makedirs(".logs", exist_ok=True)
    os.makedirs(keyword_folder / verification_dir, exist_ok=True)
    logger.add(create_logger_path(verification_dir), mode="w")

    # with shelve.open(f".cache/verification/psi4.psi4.v1.9.1.DOCS") as db:
    #     db['idx'] = 6720

    # file_path = Path("./metadata/keywords/verification/big_sample2.csv")

    # creds = [
    #     Credentials(
    #         {'author': 'scverse', 'repo': 'scanpy', 'version': '1.10.2', 'wiki': 'https://scanpy.readthedocs.io'}),
    #     Credentials({'author': 'allenai', 'repo': 'scispacy', 'version': 'v0.5.5',
    #                  'wiki': 'https://allenai.github.io/scispacy/'}),
    #     Credentials({'author': 'qutip', 'repo': 'qutip', 'version': 'v5.0.4', 'wiki': 'https://qutip.org'}),
    #     Credentials({'author': 'hail-is', 'repo': 'hail', 'version': '0.2.133', 'wiki': 'https://hail.is'}),
    # ]
    creds = credential_list

    try:
        # for file_path in keyword_folder.glob("*.csv"):
        for file_path in optimized_keyword_folder.glob("*.csv"):
            if MatchSource.CODE_COMMENT.value in file_path.stem:
                logger.info(f"Skipping CODE_COMMENTS for {file_path.stem}, as dataset is incomplete")
                continue
            if any(cred.get_ref(".") in file_path.stem for cred in creds):
                res_filepath = keyword_folder / f"{verification_dir}/{file_path.stem}.verified.csv"
                verify_file(file_path, res_filepath)  # res_filepath = file_path.with_stem("test123")
    except Exception as e:
        logger.error(e)


if __name__ == "__main__":
    main()
