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
You are a meticulous data pre-processing bot for a scientific study. Your ONLY task is to distinguish between **human-authored text** and **machine-generated artifacts**.

**Your absolute priority is the Human-Authorship Principle:** You must determine if the primary author of the text snippet is a human communicating with another human.

- **Human-Authored Text (KEEP):** Explanations, documentation, comments, questions, and discussions.
- **Machine-Generated Artifacts (ELIMINATE):** Logs, test results, build outputs, stack traces, and boilerplate notices.

**Crucial Tie-Breaker:** If a machine-generated artifact (like a compiler warning) contains a readable English sentence, the Human-Authorship Principle still applies. The text's origin is a machine, so it **MUST BE ELIMINATED**.
"""

    @classmethod
    def to_prompt(cls, x: pd.Series) -> str:
        return f"""
You are a data filtering bot. Your task is to analyze the user-provided text snippet and decide whether to keep it or eliminate it based on the Human-Authorship Principle. You must return a JSON object with a boolean `to_eliminate` field and a `reasoning` string.

## Principle Hierarchy:
1.  **Human Authorship is Paramount:** If a human wrote it to explain, discuss, or instruct, **KEEP IT**. This rule overrides all others.
2.  **Machine Generation is Noise:** If a machine generated it as a status report, **ELIMINATE IT**, even if it contains English words.
3.  **When in Doubt, KEEP:** If you cannot definitively determine the author is a machine, you must default to keeping the text.

---

### **Rule 1: Content to KEEP (Human-Authored)**
You **MUST KEEP** text written by a human. This includes:

1.  **Explanations & Rationale:** Prose that explains *what* something is, *how* it works, or *why* a decision was made.
2.  **Documentation:** Human-written descriptions of code, models, or data. This explicitly includes tables or lists that serve to document something.
3.  **Interactive Communication:** Questions, answers, bug reports, and discussions between developers (e.g., "Hi, I'm having an issue...").

---

### **Rule 2: Content to ELIMINATE (Machine-Generated or Boilerplate)**
You **MUST ELIMINATE** text that is a machine-generated artifact or standard boilerplate.

1.  **Logs, Traces, and Test Results:** Any output from a program's execution.
    *   **Crucial Test:** Was this text generated automatically by a program to report its status? If yes -> **ELIMINATE**.
    *   **Example:** A compiler warning like `warning C4244: conversion from size_t to float` -> ELIMINATE.

2.  **Lists of Raw Data:** A bare list of technical items (e.g., file paths, API names) that is NOT presented within a documentary context.
    *   **Crucial Test:** Is this a table in a README file meant to explain something? If yes -> **KEEP**. Is it just a raw list of files from a `ls` command? If yes -> **ELIMINATE**.

3.  **Boilerplate Notices:** Standard, non-project-specific text.
    *   **Includes:** Copyright notices and software licenses.

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
    