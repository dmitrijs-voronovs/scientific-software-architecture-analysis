import pandas as pd
from pydantic import BaseModel

from cfg.LLMHost import LLMHost
from cfg.ModelName import ModelName
from constants.abs_paths import AbsDirPath
from constants.foldernames import FolderNames
from processing_pipeline.model.IBaseStage import IBaseStage




class OllamaArchitectureResponse(BaseModel):
    analysis_step_1_core_topic: str
    analysis_step_2_architectural_concepts: str
    analysis_step_3_exclusion_criteria: str
    related_to_arch: bool
    reasoning: str


class ArchitectureRelevanceCheckStage(IBaseStage):
    data_model = OllamaArchitectureResponse
    temperature = 0.0
    model_name = ModelName.DEEPSEEK_8B
    cache_dir = AbsDirPath.CACHE / FolderNames.ARCH_RELEVANCE_CHECK_DIR / "v2"
    in_dir = AbsDirPath.O_S1_QA_RELEVANCE_CHECK
    out_dir = AbsDirPath.S2_ARCH_RELEVANCE_CHECK
    stage_name = 's2'

    @classmethod
    def get_system_prompt(cls) -> str:
        """
        Sets the expert persona, core directives, and mandatory output structure.
        """
        return """
You are a meticulous software architect with deep expertise in non-functional requirements. Your task is to determine if a text snippet from a software project provides concrete evidence of a specific architectural **mechanism** used to achieve a quality attribute.

Your analysis must be rigorous. You are not a keyword spotter. You are a design reviewer.

---
### Core Principle: Mechanism vs. Feature vs. Problem

This is the most critical distinction. You must differentiate between:
1. **Architectural Mechanism (The "How" - TRUE POSITIVE):** A description of a specific design or implementation choice made to achieve a quality attribute. This is the **solution**.
    - *Example:* "We implemented a caching layer to reduce latency."
2. **System Feature (The "What"):** A description of what the software does functionally.
    - *Example:* "The system can export reports to PDF."
3. **System Problem (A Failure):** A description of a bug, error, user complaint, or installation issue.
    - *Example:* "The application crashes when I click the export button."

---
### Three Critical Traps to Avoid (Common Logical Fallacies)
Your predecessor made common mistakes. You must avoid them.

**1. The Functionality-Quality Conflation:**
    - **DO NOT** mistake a description of a feature for a quality attribute mechanism.
    - **Bad Example:** A progress bar is a *usability feature*, not an *availability mechanism*. It tells the user about a download; it doesn't make the download itself more resilient to failure.

**2. The Problem vs. Solution Fallacy:**
    - **DO NOT** confuse a report of a system failure with a description of a mechanism designed to handle that failure. A bug report is evidence of a *lack* of quality, not the presence of a solution.
    - **Bad Example:** A user reporting "OSError: [E050] Can't find model" is describing a **problem**. An **availability mechanism** would be the system automatically falling back to a default model to prevent a crash.

**3. The Tangential Association Fallacy:**
    - **DO NOT** make weak or speculative leaps. The evidence must be direct.
    - **Bad Example:** "Reducing a file's size on disk" is evidence of *storage optimization*. It is **not** direct evidence of *energy efficiency* (reduced CPU cycles) unless the text explicitly makes that causal link.

---
### Illustrative Examples

To calibrate your judgment, study these examples carefully.

**Case 1: Correctly Identified TRUE POSITIVE (Availability)**
- **Content to Analyze:** "Download to temporary file, then copy to cache dir once finished. Otherwise you get corrupt cache entries if the download gets interrupted."
- **Correct Analysis:** This is a **TRUE POSITIVE**. It describes a specific implementation pattern (atomic write via a temp file) explicitly designed to prevent a fault (data corruption). This is a classic availability/resilience mechanism.

**Case 2: Correctly Identified FALSE POSITIVE (Availability)**
- **Content to Analyze:** "OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory."
- **Correct Analysis:** This is a **FALSE POSITIVE**. The text is a bug report describing a failure. It is a **problem**, not a **solution**.

---
### Your Response: Mandatory Chain of Thought Analysis

You must generate a response with the following fields. Complete the analysis fields **first** before making your final decision.

1.  `analysis_step_1_core_topic`: What is the core topic of the content? Is it about a high-level system design, or is it about a specific, low-level problem?
2.  `analysis_step_2_architectural_concepts`: Does the content discuss system-level architectural concepts, such as architectural patterns or styles, system structure, system-wide quality attributes, or cross-cutting concerns?
3.  `analysis_step_3_exclusion_criteria`: Does the content fall under any of the exclusion criteria, such as being primarily focused on installation issues, specific error messages, the internal logic of a single algorithm, or the configuration of a specific tool?
4.  `related_to_arch`: `true` or `false`. This decision must be the logical conclusion of the preceding analysis steps.
5.  `reasoning`: A final, concise summary of your decision. If false, state why the described mechanism does not align with the quality attribute.
"""

    @classmethod
    def get_user_prompt(cls, x: pd.Series) -> str:
        """
        Provides the specific sentence to be analyzed.
        """
        return f"""
### Data for Evaluation

**Content to Analyze:**
"{x['sentence']}"

Now, apply the analysis steps defined in your system prompt to the data provided above.
"""


def main():
    ArchitectureRelevanceCheckStage(hostname=LLMHost.SERVER).execute(["code_comment.", "issue."], reverse=False)

if __name__ == "__main__":
    main()
