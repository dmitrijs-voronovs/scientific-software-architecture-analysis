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
    text_summary: str
    architectural_goal_analysis: str
    tactic_evaluation: str
    selected_tactic: TacticType
    justification: str
    tactic_response: str
    response_measure: str


class TacticExtractionStage_v2(IBaseStage):
    data_model = TacticModelResponse
    temperature = 0.0
    model_name = ModelName.DEEPSEEK_8B
    cache_dir = AbsDirPath.CACHE / FolderNames.TACTIC_EXTRACTION_DIR / "v2"
    in_dir = AbsDirPath.O_S2_ARCH_RELEVANCE_CHECK
    out_dir = AbsDirPath.S3_TACTIC_EXTRACTION
    stage_name = 's3'

    @classmethod
    def get_system_prompt(cls) -> str | None:
        return f"""You are an expert software architect with a specialization in analyzing developer communications to identify design patterns and architectural tactics. Your primary goal is to analyze a given text and classify it with the single most specific architectural tactic from a provided list, and then extract the tactic's response and response measure if they are described.

You must follow a structured, step-by-step reasoning process. Your entire response must be a single, flat JSON object. Do not use nested objects or markdown.

The JSON object must contain the following fields in this exact order:
- "text_summary": A brief, neutral summary of the key information in the text.
- "architectural_goal_analysis": An analysis of the summary to determine the underlying architectural goal or problem being addressed (e.g., "improve performance," "increase flexibility," "prevent errors").
- "tactic_evaluation": A systematic evaluation of EACH available tactic from the detailed list. For each tactic, provide a brief analysis of its applicability to the text and conclude with either "Match" or "No Match".
- "selected_tactic": The single best-fitting tactic from the "Relevant Tactic Names" list provided in the user prompt. If no tactic is a strong match, you MUST select "None".
- "justification": A single, concise sentence explaining why the selected tactic is the best fit, directly linking a specific part of the original text to the tactic's definition. If "None" is selected, explain why no tactic applies.
- "tactic_response": The direct qualitative outcome or effect of applying the selected tactic. This describes WHAT changed (e.g., "execution time was reduced," "model accuracy was improved"). This field must not contain numbers or metrics. If no response is described, this must be "None".
- "response_measure": The specific, quantitative metric associated with the response. This describes HOW MUCH it changed (e.g., "from 48 minutes to 30 minutes," "F1 score improved from 94% to 98%"). If no measure is described, this must be "None".

Follow these rules strictly:
1.  Your primary objective is the final classification in "selected_tactic". All other fields are mandatory steps to reach that conclusion.
2.  The "selected_tactic" MUST be one of the names from the "Relevant Tactic Names" list, or "None". Do not select a tactic from a different category.
3.  Base your entire analysis ONLY on the provided "Text To Analyze" and "Available Tactics". Do not use external knowledge.
4.  The "tactic_response" and "response_measure" must be the direct result of the "selected_tactic". Do not extract outcomes from unrelated parts of the text.
5.  If "selected_tactic" is "None", then "justification" must explain why, and both "tactic_response" and "response_measure" MUST be "None".
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
    TacticExtractionStage_v2(hostname=LLMHost.GREEN_LAB, disable_cache=True, cot_prompt=True, n_threads_override=5, batch_size_override=10).execute([],
                                                           reverse=False)


if __name__ == "__main__":
    main()