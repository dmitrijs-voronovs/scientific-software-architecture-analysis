import pandas as pd
from pydantic import BaseModel

from cfg.LLMHost import LLMHost
from cfg.ModelName import ModelName
from constants.abs_paths import AbsDirPath
from constants.foldernames import FolderNames
from processing_pipeline.model.IBaseStage import IBaseStage


class OllamaNoiseFilteringResponse(BaseModel):
    to_eliminate: bool
    reasoning: str


class NoiseFilteringStage_v2(IBaseStage):
    data_model = OllamaNoiseFilteringResponse
    temperature = 0.0
    model_name = ModelName.DEEPSEEK_8B
    cache_dir = AbsDirPath.CACHE / FolderNames.NOISE_FILTERING_DIR
    in_dir = AbsDirPath.O2_KEYWORDS_MATCHING
    out_dir = AbsDirPath.S0_NOISE_FILTERING
    stage_name = 's0'

    @classmethod
    def get_system_prompt(cls) -> str | None:
        return """
You are a meticulous data pre-processing bot for a scientific study. Your ONLY task is to filter a dataset of text snippets, keeping human-written prose and discarding programmatic noise.

**Your absolute priority is to PRESERVE HUMAN-WRITTEN KNOWLEDGE.**

You must operate under the following core principle: **If a text snippet contains any meaningful explanation, rationale, or instruction written by a human for another human, it MUST BE KEPT, even if it is short or surrounded by code.** You are a noise filter, not a quality critic.
        """

    @classmethod
    def to_prompt(cls, x: pd.Series) -> str:
        return f"""
You are a data filtering bot. Your task is to analyze the user-provided text snippet and decide whether to keep it or eliminate it based on a strict set of rules. You must return a JSON object with a boolean `to_eliminate` field and a `reasoning` string.

## Core Mandate:
**DEFAULT TO KEEPING THE TEXT.** You must only eliminate text that **unambiguously** fits the "Eliminate" criteria. If there is any doubt, you must keep the text.

---

### **Rule 1: Content to KEEP**
You **MUST KEEP** any text that serves one of the following purposes, regardless of its length or format:

1.  **Explanation or Rationale:** The text explains *what* something is, *how* it works, or *why* a decision was made.
    *   **Includes:** Detailed documentation, simple one-sentence function descriptions, comments explaining a line of code.
    *   **Example:** "This function returns the default graphics context." -> KEEP.

2.  **Instruction or Communication:** The text represents a direct communication between developers.
    *   **Includes:** Bug reports, critiques, suggestions for future work, and action items.
    *   **Example:** "FIXME: Update the type on all intervening expressions." -> KEEP.

---

### **Rule 2: Content to ELIMINATE**
You **MUST ELIMINATE** text that is **EXCLUSIVELY** one of the following and lacks any of the explanatory or communicative elements from Rule 1:

1.  **Machine-Generated Output:** Raw program logs, stack traces, compiler errors, or test suite failures.
    *   **Crucial Test:** If there is no human analysis wrapping the log, eliminate it.

2.  **Lists of Code or Data:** A bare list of file paths, API function names, variables, or data table entries.
    *   **Crucial Test:** If the list is not part of a larger sentence or paragraph that explains its purpose, eliminate it.

3.  **Pure Code:** Executable code with no comments explaining its purpose.

4.  **Boilerplate Legal/License Text:** Standard copyright headers or license text that provides no project-specific information.

---

## Analysis Task:

Now, analyze the following text snippet and provide your JSON output.

**Content to analyze:**
{x['sentence']}
"""


def main():
    NoiseFilteringStage_v2(hostname=LLMHost.GREEN_LAB, batch_size_override=10, disable_cache=True).execute()


if __name__ == "__main__":
    main()
    