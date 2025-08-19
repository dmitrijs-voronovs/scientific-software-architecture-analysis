from typing import Dict

import pandas as pd
import yaml
from pydantic import BaseModel

from cfg.LLMHost import LLMHost
from cfg.ModelName import ModelName
from cfg.tactics.tactic_list_simplified import TacticType
from constants.abs_paths import AbsDirPath
from constants.foldernames import FolderNames
from processing_pipeline.model.IBaseStage import IBaseStage


def get_structured_tactic_list_by_qa() -> Dict[str, str]:
    """
    Loads tactic definitions from YAML and formats them into a string for each quality attribute.
    """
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


def get_structured_tactic_names_by_qa() -> Dict[str, str]:
    """
    Loads tactic names from YAML and formats them into a comma-separated string for each quality attribute.
    """
    with open(AbsDirPath.TACTICS / "tactic_list_modified.yaml", "r") as f:
        tactics = yaml.safe_load(f)
    tactics_names_by_qa = {}

    for attr in tactics['tactics']:
        tactic_names = []
        for category in attr['tactic_categories']:
            for tactic in category['tactics']:
                tactic_names.append(tactic['name'])

        tactics_names_by_qa[attr['quality_attribute'].lower()] = ", ".join(tactic_names)

    return tactics_names_by_qa


# This dictionary translates the QA found on an item
# to the QA that has a corresponding tactic list.
QA_TO_TACTIC_MAP = {'deployability': 'modifiability',  # mapped
                    'integrability': 'interoperability',  # mapped
                    'reliability': 'availability',  # mapped
                    }

# Global maps for tactic details and names
tactics_by_qa_map = get_structured_tactic_list_by_qa()
tactic_names_by_qa_map = get_structured_tactic_names_by_qa()


def get_tactic_list_for_qa(qa: str) -> str:
    """
    Retrieves the detailed tactic definitions for a given quality attribute.
    """
    mapped_qa = QA_TO_TACTIC_MAP.get(qa, qa)
    tactic_list = tactics_by_qa_map.get(mapped_qa)
    assert tactic_list is not None, f"No tactic list found for qa {qa}"
    return tactic_list


def get_tactic_names_for_qa(qa: str) -> str:
    """
    Retrieves the comma-separated tactic names for a given quality attribute.
    """
    mapped_qa = QA_TO_TACTIC_MAP.get(qa, qa)
    tactic_names = tactic_names_by_qa_map.get(mapped_qa)
    assert tactic_names is not None, f"No tactic names found for qa {qa}"
    return tactic_names


class TacticModelResponse(BaseModel):
    """Pydantic model for the multi-stage filtering and semantic analysis JSON output."""
    architectural_activity_extraction: str
    core_concept_analysis: str
    is_tactic_relevant: bool
    relevance_reason: str
    tactic_evaluation: str
    selected_tactic: TacticType
    justification: str


class TacticExtractionStage_v2(IBaseStage):
    data_model = TacticModelResponse
    temperature = 0.0
    model_name = ModelName.DEEPSEEK_1_5B
    cache_dir = AbsDirPath.CACHE / FolderNames.TACTIC_EXTRACTION_DIR / "v2"
    in_dir = AbsDirPath.O_S2_ARCH_RELEVANCE_CHECK
    out_dir = AbsDirPath.S3_TACTIC_EXTRACTION
    stage_name = 's3'

    @classmethod
    def get_system_prompt(cls) -> str | None:
        return """You are an expert software architect with a specialization in analyzing developer communications to identify design patterns and architectural tactics. Your primary goal is to meticulously filter and analyze a given text to determine if it describes a concrete architectural tactic.

You must follow a strict, sequential reasoning process. Your entire response must be a single, flat JSON object. Do not use nested objects or markdown.

The JSON object must contain the following fields in this exact order:
- "architectural_activity_extraction": First, quote the exact sentence(s) from the text that describe a concrete technical change, an implemented solution, or a deliberate design decision. If no such sentences exist, you must state "No concrete architectural activity described."
- "core_concept_analysis": Based ONLY on the extracted sentences, summarize the single primary architectural concept in one sentence. If no activity was extracted, this must be "None".
- "is_tactic_relevant": Based on the core concept, answer with 'true' or 'false' to the question: "Does this concept describe a deliberate design decision intended to influence a quality attribute?".
- "relevance_reason": Briefly explain your reasoning for the 'is_tactic_relevant' decision. If the concept is just a bug fix, user question, or documentation, the answer must be false.
- "tactic_evaluation": IF AND ONLY IF 'is_tactic_relevant' is true, systematically evaluate each available tactic against the 'core_concept_analysis'. Otherwise, state "Not applicable due to relevance check failure."
- "selected_tactic": The single best-fitting tactic from the "Relevant Tactic Names" list. If 'is_tactic_relevant' is false, **or if no tactic from the provided list is a good semantic match**, this MUST be "None".
- "justification": If a tactic is selected, explain why it is the best semantic fit for the 'core_concept_analysis'. If "None" is selected, use the 'relevance_reason' **or the result of the tactic evaluation** to explain why.

Follow these rules strictly:
1.  Your primary objective is to correctly filter the text. The classification is secondary.
2.  If 'is_tactic_relevant' is false, you MUST stop the analysis and set 'selected_tactic' to "None".
3.  The 'selected_tactic' must be one of the names from the "Relevant Tactic Names" list, or "None".
4.  Base your entire analysis ONLY on the provided "Text To Analyze". Do not use external knowledge.
"""

    @classmethod
    def to_prompt(cls, x: pd.Series) -> str:
        tactic_definitions = get_tactic_list_for_qa(x['qa'])
        tactic_names_list = get_tactic_names_for_qa(x['qa'])

        return f"""
Based on the rules provided in the system prompt, analyze the following available tactics and text and provide the JSON output.

---

## Relevant Tactic Names for this Quality Attribute
{tactic_names_list}

---

## Available Tactics (with definitions)
{tactic_definitions}

---
## Text To Analyze:
"{x['sentence']}"
"""


def main():
    TacticExtractionStage_v2(hostname=LLMHost.SERVER, n_threads_override=8, batch_size_override=16).execute()

if __name__ == "__main__":
    main()
