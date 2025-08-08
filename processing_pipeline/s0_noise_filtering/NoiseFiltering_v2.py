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

## Core Mandate & Litmus Test

Before applying the rules, perform this litmus test: **"Was this text written by a human to explain something to another human?"**
- If the answer is YES, you **MUST KEEP** the text.
- If the answer is NO, it is likely machine-generated noise that should be eliminated.

**Your default action is to KEEP.** You must only eliminate text that is unambiguously machine-generated.

---

### **Rule 1: Content to KEEP (Human-Authored)**
You **MUST KEEP** text written by a human. This includes:

1.  **Explanations & Documentation (of ANY length):** Prose that explains *what* something is, *how* it works, or *why* a decision was made.
    *   **CRITICAL:** This is the most important rule. A short, single-sentence function description or code comment (e.g., "Initializes a checkpoint manager.") is high-value human knowledge and **MUST BE KEPT**. Do not mistake brevity for being machine-generated.

2.  **Interactive Communication:** Questions, answers, bug reports, issue discussions, and developer comments. (e.g., "Hi, I'm having an issue...").

3.  **Documentation Containing Code/Data:** Human-written prose that includes code snippets, tables, or lists as examples. The primary signal is the human-written explanation surrounding these elements.

---

### **Rule 2: Content to ELIMINATE (Machine-Generated or Boilerplate)**
You **MUST ELIMINATE** text that is clearly a machine-generated artifact or standard boilerplate, AND which is not part of a larger human-authored explanation.

1.  **Logs, Traces, and Test Reports:** Any output from a program's execution, including build logs, test suite results, stack traces, and compiler errors.
    *   **Crucial Test:** Was this text generated *automatically* by a program to report its status? If yes -> **ELIMINATE**.

2.  **Raw Data Lists:** A bare list of technical items (e.g., file paths, API names) that is **NOT** part of a larger documentary context.
    *   **Crucial Test:** Is this a table in a README file meant to explain something? If yes -> **KEEP**. Is it just a raw list of files from an `ls` command? If yes -> **ELIMINATE**.

3.  **Boilerplate Notices:** Standard, non-project-specific text such as copyright notices and software licenses.

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
    