from functools import cache
from typing import Dict

import pandas as pd
import yaml

from cfg.LLMHost import LLMHost
from cfg.ModelName import ModelName
from cfg.tactics.tactic_list_simplified import TacticSimplifiedModelResponse
from constants.abs_paths import AbsDirPath
from constants.foldernames import FolderNames
from processing_pipeline.model.IBaseStage import IBaseStage


def get_structured_tactic_list_by_qa() -> Dict[str, str]:
    with open(AbsDirPath.TACTICS / "tactic_list_modified.yaml", "r") as f:
        tactics = yaml.safe_load(f)
    tactics_by_qa = {}

    for attr in tactics['tactics']:
        prompt_lines = []
        for category in attr['tactic_categories']:
            prompt_lines.append(f"\n#### {category['category_name']}")
            for tactic in category['tactics']:
                prompt_lines.append(f"- **{tactic['name']}**: {tactic['description']}")

        tactics_by_qa[attr['quality_attribute'].lower()] = "\n".join(prompt_lines)

    return tactics_by_qa

# This dictionary translates the QA found on an item
# to the QA that has a corresponding tactic list.
QA_TO_TACTIC_MAP = {
    'deployability': 'modifiability',         # mapped
    'integrability': 'interoperability',      # mapped
    'reliability': 'availability',          # mapped
}

tactics_by_qa_map = get_structured_tactic_list_by_qa()

def get_tactic_list_for_qa(qa: str) -> str:
    mapped_qa = QA_TO_TACTIC_MAP.get(qa, qa)
    tactic_list = tactics_by_qa_map.get(mapped_qa)
    assert tactic_list is not None, f"No tactic list found for qa {qa}"
    return tactic_list

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
You are an expert in software architecture tactics. Your task is to analyze user-provided text and identify the single most specific software architecture tactic being described from a list I will provide.

## Guiding Principles
- Focus on the Mechanism: Identify the architectural *how* (the solution or feature), not the *why* (the benefit).
- Handle Non-Feature Descriptions: If the text is a user question, bug report, installation issue, or a general discussion *about* the software rather than a description of a feature *within* the software, you **must** classify the tactic as `None`.

## Your Task
You will be given a list of "Available Tactics" and a "Text to Analyze". Based on these, provide a single JSON object with two fields:
1.  `tactic`: The name of the single most specific tactic you identified from the provided list, or `None`.
2.  `response`: A one-sentence summary of the functionality described, starting with "The system...". If the tactic is `None`, summarize the user's query or the text's purpose.

---
### Example of Handling a Non-Feature
- Text: "Tensorflow version of the model checkpoint; What is the version of tensorflow for generating the checkpoint files (`index`, `meta`, `data`)? And is there any way that I can load these checkpoints into a standalone tensorflow program and then dump it as a `.onnx` file?"
  - `tactic`: None
  - `response`: The system is being asked about its TensorFlow version and how to convert its model checkpoints to another format.
---
You will now be provided with the list of available tactics and a text to analyze. Apply these rules to the text that follows.
"""

    @classmethod
    def to_prompt(cls, x: pd.Series) -> str:
        return f"""
Based on the rules provided, analyze the following available tactics and text and provide the JSON output.

---

## Available Tactics
{get_tactic_list_for_qa(x['qa'])}

---
## Text To Analyze:
"{x['sentence']}"
"""


def main():
    TacticExtractionStage(hostname=LLMHost.SERVER).execute(["issue.", "docs"], reverse=False)
    # TacticExtractionStage(hostname=LLMHost.GREEN_LAB, batch_size_override=5, n_threads_override=1,
    #                       model_name_override=ModelName.DEEPSEEK_8B, disable_cache=True).execute()


if __name__ == "__main__":
    main()
