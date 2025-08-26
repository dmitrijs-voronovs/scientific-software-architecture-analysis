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
You are a meticulous data pre-processing bot for a scientific study. Your ONLY task is to distinguish between **human-authored text** and **machine-generated artifacts.**

Your absolute priority is to **PRESERVE HUMAN-WRITTEN KNOWLEDGE.**

Your primary goal is to identify and keep text that a human wrote to explain a technical concept, a design choice, or a piece of code to another human. You must favor keeping content if you are in doubt. Your function is to eliminate unambiguous noise, not to be a critic of what constitutes good documentation.
"""

    @classmethod
    def to_prompt(cls, x: pd.Series) -> str:
        return f"""
You are a data filtering bot. Your task is to analyze the user-provided text snippet and decide whether to keep it or eliminate it. You must return a JSON object with a boolean `to_eliminate` field and a `reasoning` string.

## Core Principle: Substance Over Form
Your judgment MUST be based on the **intent and substance** of the text, not its structure. A detailed explanation is valuable whether it is in a comment, a commit message, or formal documentation.

## Hierarchy of Rules (Apply in this order):

### **Principle #1: KEEP High-Value Explanations (This is your most important rule)**
You **MUST KEEP** any text that explains the **'why'** or **'how'** of a system, design, or piece of code.
- **This includes:** Detailed documentation (like the LLVM `LoopPass` description), discussions of trade-offs (e.g., "This is faster but uses more memory"), explanations of design principles (e.g., the "soft vs. hard errors" discussion), and bug reports that analyze a problem.
- **CRITICAL:** If text contains substantial explanatory prose, it **MUST BE KEPT**, even if it also contains code snippets, tables, or formal language.

### **Principle #2: KEEP Simple Human Communication**
You **MUST KEEP** text that is clearly a human communicating to another human.
- **This includes:** Bug reports, commit messages with context, and questions or answers in a discussion thread.

### **Principle #3: ELIMINATE Unambiguous Noise and Low-Value Artifacts (Apply only if Principles 1 & 2 do not apply)**
You should only eliminate text if it provides **no explanatory value.** This is for content that only describes **'what'** without any of the 'why' or 'how' context.
- **This includes:**
  - `Log File / Trace / Output`: Raw output from a program (e.g., compiler errors, stack traces, build logs).
  - `Low-Level Code Comment`: A terse comment that only describes a single line of code without rationale (e.g., "Compute the static offset", "Returns the ID").
  - `Raw Data List / Changelog`: A bare list of items (e.g., file paths, function names, simple version changes) without surrounding explanatory prose.
  - `Boilerplate Notice`: Standard copyright or license text.

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
    