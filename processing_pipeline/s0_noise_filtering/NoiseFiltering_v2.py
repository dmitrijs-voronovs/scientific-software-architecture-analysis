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

**Your judgment must be based on two absolute priorities:**
1.  **The Human-Authorship Principle:** Determine if the primary author is a human communicating with another human.
2.  **The Documentation Principle:** Recognize that formal technical documentation, API references, READMEs, and scientific explanations are high-value, human-authored content that MUST be preserved.

**Crucial Tie-Breaker:** The *functional category* of the content is more important than its grammatical structure.
- If a snippet is functionally a **Log** or **Boilerplate License**, it **MUST BE ELIMINATED**, even if it contains English prose.
- Conversely, if a snippet is functionally a human-written **Guide**, **Tutorial**, or **API Documentation**, it **MUST BE KEPT**, even if it is highly structured or contains many code snippets.
"""

    # In your NoiseFilteringStage class
    @classmethod
    def to_prompt(cls, x: pd.Series) -> str:
        return f"""
You are a data filtering bot. Your task is to analyze the user-provided text snippet and decide whether to keep it or eliminate it based on the principles from your system prompt. You must return a JSON object with a boolean `to_eliminate` field and a `reasoning` string.

## Core Mandate & Litmus Test

Before applying the rules, perform this litmus test: **"Was this text written by a human to explain something to another human?"**
- If the answer is YES, you **MUST KEEP** the text.
- If the answer is NO, it is a machine-generated artifact and **MUST BE ELIMINATED**.

---

### **Rule 1: Content to KEEP (Human-Authored)**
You **MUST KEEP** text if its primary purpose is human-to-human communication or documentation. This includes:

1.  **Explanations, Documentation & Scientific Prose:** Prose that explains *what* something is, *how* it works, or *why* a decision was made. This is the highest priority rule.
    *   **Includes:** Project READMEs, user guides, detailed comments, and formal academic or mathematical explanations.
    *   **CRITICAL:** Do not mistake structure, formality, or technical terms for being machine-generated. A detailed, structured document is high-value human knowledge.

2.  **API Reference & Help Text:** Docstrings and command-line help text that describe a function, its parameters, its behavior, or what it returns.
    *   **Includes:** Short, single-sentence function descriptions (e.g., "Build tfidf vectorizer and ann index.") and detailed, structured parameter lists (e.g., "--phred64 indicate the input is using phred64 scoring..."). Both **MUST BE KEPT**.

3.  **Interactive Communication:** Questions, answers, bug reports, and developer discussions.
    *   **Crucial Test:** Is this a log of a terminal session where the vast majority of text is machine output? If yes, it is a **Log** -> **ELIMINATE**.

---

### **Rule 2: Content to ELIMINATE (Machine-Generated or Boilerplate)**
You **MUST ELIMINATE** text that is clearly a machine-generated artifact or standard boilerplate.

1.  **Logs, Traces, and Test Reports:** Any output from a program's execution.
    *   **Crucial Test:** Was this text generated *automatically* by a program to report its status? If yes -> **ELIMINATE**.

2.  **Raw Data Lists:** A list of technical items (e.g., file paths, API names, chemical names) that is **NOT** explained by surrounding human-written sentences.
    *   **Crucial Test:** Is this a table in a README file with a caption? If yes -> **KEEP**. Is it just a raw list of terms or files without an explanatory sentence? If yes -> **ELIMINATE**.

3.  **Boilerplate Notices:** Standard, non-project-specific legal or copyright text.
    *   **Example:** "Copyright 2017 Google LLC..." -> **ELIMINATE**.

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
    