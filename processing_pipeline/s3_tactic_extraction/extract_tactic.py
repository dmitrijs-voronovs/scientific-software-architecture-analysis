import math
import os
import shelve
import signal
import sys
import traceback
from pathlib import Path
from typing import List

import dotenv
import pandas as pd
from langchain_ollama import ChatOllama
from loguru import logger
from tenacity import RetryError
from tqdm import tqdm

from cfg.tactic_description_full import tactic_descriptions_full
from cfg.tactic_list_simplified import TacticSimplifiedModel
from constants.foldernames import FolderNames
from cfg.repo_credentials import selected_credentials
from utils.utils import create_logger_path

# Load environment variables from .env file
dotenv.load_dotenv()


def request_ollama_chain(prompts: List[str], base_url: str) -> List[TacticSimplifiedModel]:
    # model_name = "gemma"
    # model_name = "gemma2"
    model_name = "deepseek-r1:8b"
    model = ChatOllama(model=model_name, base_url=base_url, format=TacticSimplifiedModel.model_json_schema())
    batch_answers = model.batch(prompts)
    return [TacticSimplifiedModel.model_validate_json(answer.content) for answer in batch_answers]


tactic_descriptions_list = "\n".join(
    f"- {tactic}: (quality attribute '{details["quality_attribute"]}', category '{details["tactic_category"]}') {details["description"]}"
    for tactic, details in tactic_descriptions_full.items())

tactic_descriptions_list_simplified = "\n".join(
    f"- {tactic}: {details["description"]}"
    for tactic, details in tactic_descriptions_full.items())

tactic_prompt_v1 = lambda x: f"""
You are an expert in evaluating and categorizing architecture tactics in software engineering. You possess the necessary skills to categorize text according to software architecture tactics, quality attributes, and responses.

Given a piece of text related to software architecture, your task is to:
1. Identify the specific tactic being described
2. Provide a clear "response" which in the context of software architecture refers to the activity undertaken by the system (for runtime qualities) or the developers (for development-time qualities) as a result of the arrival of a stimulus

Analyze the following text:
{x["sentence"]}

Concept of Tactic, Quality Attribute, and Response:
- An architectural tactic is a design decision that directly affects a system's response to a stimulus, influencing the achievement of a quality attribute. The primary purpose of tactics is to achieve desired quality attributes by imparting specific qualities to a design.
- The concept of "response" is central to this relationship. When a stimulus occurs, quality attribute requirements define the desired response. Tactics are employed to control these responses, ensuring the system exhibits behavior that satisfies particular quality attribute requirements.
- Different tactics can achieve different quality attributes, sometimes with multiple tactics improving a single quality attribute. Architectural patterns can be viewed as "packages" of tactics that work together to address recurring design problems.

Available Quality Attributes:
- Availability: the ability of a system to be available for use, particularly by masking or repairing faults to minimize service outage.
- Interoperability: the ability of a system to exchange information and function with other systems in a shared environment.
- Modifiability: the ease with which changes can be made to a system to accommodate new features, adapt to new environments, or fix bugs.
- Performance: concerns the timing behavior of a system and its ability to meet timing requirements in response to events.
- Security: the degree to which a system protects information and data from unauthorized access and manipulation, ensuring confidentiality, integrity, and availability.
- Testability: the ease with which software can be made to demonstrate its faults through testing.
- Usability: describes how easy it is for users to accomplish desired tasks with effectiveness, efficiency, and satisfaction.
- Energy Efficiency: relates to the minimization of energy consumption by the software system and its underlying hardware.

Tactic descriptions:
{tactic_descriptions_list}

Examples:
- Availability:
    Stimulus: Server becomes unresponsive.
    Tactic: Heartbeat Monitor (Detect Faults).
    Response: Inform Operator, Continue to Operate.
    Response Measure: No Downtime.
- Performance:
    Stimulus: Users initiate transactions.
    Tactic: Increase Resources (Manage Resources).
    Response: Transactions Are Processed.
    Response Measure: Average Latency of Two Seconds.
- Security:
    Stimulus: Disgruntled employee attempts to modify the pay rate table.
    Tactic: Maintain Audit Trail (React to Attacks).
    Response: Record attempted modification.
    Response Measure: Time taken to restore data.
- Testability:
    Stimulus: Need to test a specific unit of code.
    Tactic: Specialized Interfaces (Control and Observe System State).
    Response: System can be controlled to perform desired tests and results can be observed.
    Response Measure: Effort involved in finding a fault.
- Usability:
    Stimulus: User interacts with the system and makes an error.
    Tactic: Undo (Support User Initiative).
    Response: Ability to reverse the incorrect action.
    Response Measure: Number of errors made by the user, amount of time or data lost when an error occurs.

Instructions:
1. Carefully analyze the text to determine which quality attribute, tactic category it most closely relates to.
2. Determine the specific tactic being described.
3. Provide a clear description of the system's response to the stimulus described in the text.
"""


tactic_prompt_v2 = lambda x: f"""
You are an expert in software architecture tactics. Your task is to analyze the given text and categorize it according to software architecture tactics, quality attributes, and system responses.

### **Key Concepts**
- **Tactics**: Design decisions that influence system responses to achieve quality attributes.
- **Quality Attributes**: Characteristics such as performance, security, modifiability, etc.
- **Response**: The system's reaction to a stimulus, either at runtime or development time.

### **Available Quality Attributes**
- **Availability**: Minimizing service outages by masking or repairing faults.
- **Interoperability**: Enabling seamless data exchange between systems.
- **Modifiability**: Facilitating easy changes for new features, bug fixes, or adaptations.
- **Performance**: Ensuring the system meets timing and throughput requirements.
- **Security**: Protecting data from unauthorized access and ensuring confidentiality.
- **Testability**: Making software easy to test and debug.
- **Usability**: Enhancing user experience and reducing operational errors.
- **Energy Efficiency**: Reducing energy consumption in software and hardware.

### **Your Task**
For the given text:
1. Identify the **quality attribute** it addresses.
2. Determine the **specific tactic** from the list below.
3. Describe the **system's response** to the stimulus.

### **Examples**
- **Availability**  
  - **Stimulus**: Server becomes unresponsive.  
  - **Tactic**: Heartbeat Monitor (Detect Faults).  
  - **Response**: Inform Operator, Continue to Operate.  
  - **Response Measure**: No Downtime.  

- **Performance**  
  - **Stimulus**: Users initiate transactions.  
  - **Tactic**: Increase Resources (Manage Resources).  
  - **Response**: Transactions Are Processed.  
  - **Response Measure**: Average Latency of Two Seconds.  

- **Security**  
  - **Stimulus**: Disgruntled employee attempts to modify the pay rate table.  
  - **Tactic**: Maintain Audit Trail (React to Attacks).  
  - **Response**: Record attempted modification.  
  - **Response Measure**: Time taken to restore data.  

### **Available Tactics**
{tactic_descriptions_list}

### **Analyze the Following Text**
"{x['sentence']}"
"""

tactic_prompt = lambda x: f"""
You are an expert in software architecture tactics. Your task is to analyze the given text and categorize it according to software architecture tactics, quality attributes, and system responses.

### **Key Concepts**
- **Tactics**: Design decisions that influence system responses to achieve quality attributes.
- **Quality Attributes**: Characteristics such as performance, security, modifiability, etc.
- **Response**: The system's reaction to a stimulus, either at runtime or development time.

### **Available Quality Attributes**
- **Availability**: Minimizing service outages by masking or repairing faults.
- **Interoperability**: Enabling seamless data exchange between systems.
- **Modifiability**: Facilitating easy changes for new features, bug fixes, or adaptations.
- **Performance**: Ensuring the system meets timing and throughput requirements.
- **Security**: Protecting data from unauthorized access and ensuring confidentiality.
- **Testability**: Making software easy to test and debug.
- **Usability**: Enhancing user experience and reducing operational errors.
- **Energy Efficiency**: Reducing energy consumption in software and hardware.

### **Your Task**
For the given text:
1. Identify the **quality attribute** it addresses.
2. Determine the **specific tactic** from the list below.
3. Describe the **system's response** to the stimulus.

### **Examples**
- **Availability**  
  - **Stimulus**: Server becomes unresponsive.  
  - **Tactic**: Heartbeat Monitor (Detect Faults).  
  - **Response**: Inform Operator, Continue to Operate.  
  - **Response Measure**: No Downtime.  

- **Performance**  
  - **Stimulus**: Users initiate transactions.  
  - **Tactic**: Increase Resources (Manage Resources).  
  - **Response**: Transactions Are Processed.  
  - **Response Measure**: Average Latency of Two Seconds.  

- **Security**  
  - **Stimulus**: Disgruntled employee attempts to modify the pay rate table.  
  - **Tactic**: Maintain Audit Trail (React to Attacks).  
  - **Response**: Record attempted modification.  
  - **Response Measure**: Time taken to restore data.  

### **Available Tactics**
{tactic_descriptions_list_simplified}

### **Analyze the Following Text**
"{x['sentence']}"
"""

def cleanup_and_exit(signal_num, frame):
    print("Caught interrupt, cleaning up...")
    sys.exit(0)  # Triggers the context manager's cleanup


# Register the signal handler
signal.signal(signal.SIGINT, cleanup_and_exit)


def verify_file_batched_llm(file_path: Path, res_filepath: Path, host: str, batch_size=10):
    os.makedirs(f".cache/{FolderNames.ARCHITECTURE_TACTICS}/", exist_ok=True)
    with shelve.open(f".cache/{FolderNames.ARCHITECTURE_TACTICS}/{file_path.stem}") as db:
        if db.get("processed", False):
            logger.info(f"File {file_path.stem} already processed")
            return
        logger.info(f"Processing {file_path.stem}")

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            logger.info(e)
            return

        df = df[df["related_to_architecture"] == True]
        last_idx = db.get("idx", 0)
        df = df.iloc[last_idx:].copy()
        if last_idx > 0:
            logger.info(f"Continuing from {last_idx}")
            res_filepath = res_filepath.with_suffix(f".from_{last_idx}.csv")

        df['tactic_prompt'] = df.apply(lambda x: tactic_prompt(x), axis=1)
        res = []

        for i in tqdm(range(0, len(df), batch_size), total=math.ceil(len(df) / batch_size),
                      desc=f"Verifying {file_path.stem} in batches of {batch_size}"):
            batch_df = df.iloc[i:i + batch_size]
            prompts = batch_df['tactic_prompt'].tolist()

            try:
                responses = request_ollama_chain(prompts, host)  # New batch query
                processed_responses = [((full_tactic := tactic_descriptions_full[r.tactic])["quality_attribute"],
                                        full_tactic["tactic_category"], r.tactic, full_tactic["description"],
                                        r.response) for r in responses]
                res.extend(processed_responses)
            except RetryError as error:
                logger.error(f"Retry error at batch starting index {last_idx + i}, {error}")
                responses = [(None, None, None, None, str(error))] * len(batch_df)
                res.extend(responses)
            except Exception as e:
                logger.error(e)
                errors_for_termination = ["HTTPConnectionPool",
                                          "No connection could be made because the target machine actively refused it"]
                if any(error in str(e) for error in errors_for_termination):
                    logger.error("HTTPConnectionPool error, exiting")
                    exit(1)
                responses = [(None, None, None, None, str(e))] * len(batch_df)
                res.extend(responses)

            df_to_save = df.iloc[:i + batch_size].copy()
            df_to_save['arch_quality_attribute'], df_to_save['arch_tactic_category'], df_to_save['arch_tactic'], \
            df_to_save['arch_tactic_description'], df_to_save['arch_response'] = zip(*res)
            df_to_save.to_csv(res_filepath, index=False)
            db["idx"] = last_idx + i + batch_size

        db['processed'] = True
        logger.info(f"Processed {file_path.stem}")


def extract_tactics(host, only_files_containing_text: List[str] | None = None, reverse: bool = False):
    only_files_containing_text = only_files_containing_text or []
    keyword_folder = Path("metadata/keywords/")
    optimized_keyword_folder = keyword_folder / FolderNames.ARCHITECTURE_VERIFICATION_DIR
    os.makedirs("../../.logs", exist_ok=True)
    os.makedirs(keyword_folder / FolderNames.ARCHITECTURE_TACTICS, exist_ok=True)
    logger.add(create_logger_path(FolderNames.ARCHITECTURE_TACTICS), mode="w")

    try:
        for file_path in optimized_keyword_folder.glob("*.csv"):
            if any(cred.get_ref(".") in file_path.stem for cred in (selected_credentials)):
                keep_processing = len(only_files_containing_text) == 0 or any(
                    text_to_test in file_path.stem for text_to_test in only_files_containing_text)
                if keep_processing == reverse:
                    continue

                res_filepath = keyword_folder / f"{FolderNames.ARCHITECTURE_TACTICS}/{file_path.stem}.tactics.csv"
                verify_file_batched_llm(file_path, res_filepath, host,
                                        10)  # res_filepath = file_path.with_stem("test123")
    except Exception as e:
        logger.error(f"{e}, \n{traceback.format_exc()}")


LOCAL_LLM_HOST = "http://localhost:11435"

if __name__ == "__main__":
    extract_tactics(LOCAL_LLM_HOST, ["root-project.root.v6-32-06"] )
