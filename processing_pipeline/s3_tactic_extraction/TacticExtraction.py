from functools import cache

import pandas as pd
import yaml

from cfg.LLMHost import LLMHost
from cfg.ModelName import ModelName
from cfg.tactics.tactic_list_simplified import TacticSimplifiedModelResponse
from constants.abs_paths import AbsDirPath
from constants.foldernames import FolderNames
from processing_pipeline.model.IBaseStage import IBaseStage


def get_structured_tactic_list() -> str:
    with open(AbsDirPath.TACTICS / "tactic_list.yaml", "r") as f:
        tactics = yaml.safe_load(f)
    prompt_lines = []

    for attr in tactics['tactics']:
        prompt_lines.append(f"\n\n### {attr['quality_attribute']}")
        for category in attr['tactic_categories']:
            prompt_lines.append(f"\n#### {category['category_name']}")
            for tactic in category['tactics']:
                prompt_lines.append(f"- **{tactic['name']}**: {tactic['description']}")
    return "\n".join(prompt_lines)

tactic_list = get_structured_tactic_list()

class TacticExtractionStage(IBaseStage):
    data_model = TacticSimplifiedModelResponse
    temperature = 0.0
    model_name = ModelName.DEEPSEEK_8B
    cache_dir = AbsDirPath.CACHE / FolderNames.TACTIC_EXTRACTION_DIR
    in_dir = AbsDirPath.O_S2_ARCH_RELEVANCE_CHECK
    out_dir = AbsDirPath.S3_TACTIC_EXTRACTION
    stage_name = 's3'

    @classmethod
    def get_system_prompt(cls) -> str | None:
        return f"""
You are an expert in software architecture tactics. Your task is to analyze the provided text and identify the single most specific software architecture tactic being described.

## Guiding Principles
- Focus on the Mechanism: Identify the architectural *how* (the solution or feature), not the *why* (the benefit). For example, a request to add a configuration option is a Modifiability tactic, even if it is meant to prevent an error.
- Handle Non-Feature Descriptions: If the text is a user question, bug report, installation issue, or a general discussion *about* the software rather than a description of a feature *within* the software, you **must** classify the tactic as `None`.

## Reasoning Process
To ensure accuracy, you must follow these four steps:
0.  Summarize the Core Action: In one sentence, what is the system *doing* or what functional feature is being *added* or *described*? If no feature is described, state that it is a user question or discussion.
1.  Identify Quality Attribute: Based on your summary, determine which primary Quality Attribute the text is addressing (e.g., Performance, Modifiability). If no feature is described, this is `None`.
2.  Identify Tactic Category: Within that attribute, determine the most relevant Tactic Category (e.g., Manage Resources, Reduce Coupling).
3.  Select Specific Tactic: From that category, select the single most specific tactic that best describes the action in the text.

## Your Task
Based on your reasoning, provide the following two fields:

1.  `tactic`: The name of the single most specific tactic you identified, or `None`.
2.  `response`: A one-sentence summary of the functionality or behavior described in the text, from the system's perspective. Start the sentence with "The system...". If the tactic is `None`, summarize the user's query or the nature of the text.

---
## Examples
- Text: "...for parallel processing of FASTQ files (i.e. alignment in parallel), `fastp` supports splitting the output into multiple files."
  - `tactic`: Introduce Concurrency
  - `response`: The system processes different streams of events in parallel to reduce blocked time.

- Text: "Request is a for a now CLI arg --umi_join that will define the character placed between the UMIs in read1 and read2."
  - `tactic`: Tailor Interface
  - `response`: The system adds a capability to an interface, allowing users to customize the UMI delimiter without changing the core code.

- Text: "The option `--dup_calc_accuracy` can be used to specify the level (1 ~ 6). The higher level means more memory usage and more running time."
  - `tactic`: Increase Resources
  - `response`: The system uses additional memory and processing time to reduce latency and improve calculation accuracy.
      
- Text: "Tensorflow version of the model checkpoint; What is the version of tensorflow for generating the checkpoint files (`index`, `meta`, `data`)? And is there any way that I can load these checkpoints into a standalone tensorflow program and then dump it as a `.onnx` file?"
  - `tactic`: None
  - `response`: The system is being asked about its TensorFlow version and how to convert its model checkpoints to another format.

---
## Available Tactics
{tactic_list}

---
You will now be provided with text to analyze. Apply these rules to the text that follows.
"""

    @classmethod
    def to_prompt(cls, x: pd.Series) -> str:
        return f"""
Based on the rules and tactics provided, analyze the following text and provide the JSON output.

"{x['sentence']}"
"""


def main():
    TacticExtractionStage(hostname=LLMHost.SERVER).execute(["issue", "issue_comment"], reverse=True)
    # TacticExtractionStage(hostname=LLMHost.RADU_SERVER).execute(["issue", "issue_comment"], reverse=True)
    # TacticExtractionStage(hostname=LLMHost.GREEN_LAB, batch_size_override=5, n_threads_override=1, model_name_override=ModelName.DEEPSEEK_8B, disable_cache=True).execute(["issue", "issue_comment"], reverse=True)

if __name__ == "__main__":
    main()
