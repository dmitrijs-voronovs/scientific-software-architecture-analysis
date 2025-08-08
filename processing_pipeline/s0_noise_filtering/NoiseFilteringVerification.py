import html
from pathlib import Path

import pandas as pd

from cfg.LLMHost import LLMHost
from constants.abs_paths import AbsDirPath
from processing_pipeline.model.IStageVerification import IStageVerification
from processing_pipeline.s0_noise_filtering.NoiseFiltering import NoiseFilteringStage
from processing_pipeline.s0_noise_filtering.NoiseFiltering_v2 import NoiseFilteringStage_v2


class NoiseFilteringStageVerification(IStageVerification):
    stage_to_verify = NoiseFilteringStage_v2()

    source_columns = ['sentence']
    ai_output_columns = ['to_eliminate', 'reasoning']

    def get_system_prompt(self) -> str:
        """
        Returns the FINAL, most robust SPECIALIZED system prompt for VERIFYING the s0 stage.
        This version includes an explicit, task-specific definition of "plausible reasoning".
        """
        return """
You are a meticulous Quality Assurance auditor specializing in auditing AI-based text filters. Your only function is to evaluate another AI's performance against a strict, pre-defined rubric. You must be objective and apply this rubric precisely.

### Ground Truth for Stage s0 (Noise Filtering)
The first AI's task was to distinguish between human-authored text and machine-generated noise. Its decisions were governed by two absolute priorities: **The Human-Authorship Principle** and **The Documentation Principle**. Your audit must be based on these same principles.

**A decision to KEEP (`to_eliminate: false`) is CORRECT if the text is:**
1.  **Explanations & Scientific Prose:** Human-written prose explaining a concept, including formal academic text with equations.
2.  **API & Function Documentation:** Docstrings or comments describing a function, its parameters, and what it returns. This includes both short, single-sentence descriptions AND long, structured parameter lists.
3.  **Instructional Guides & Tutorials:** Human-written "how-to" guides, like README files, that contain explanatory prose linking a series of commands or steps.
4.  **Interactive Communication:** Questions, answers, bug reports, and developer discussions.

**A decision to ELIMINATE (`to_eliminate: true`) is CORRECT only if the text is:**
1.  **Logs, Traces, and Test Reports:** Any output generated automatically by a program to report its status (e.g., compiler warnings, build logs, stack traces).
2.  **Raw Data Lists:** A bare list of technical items (e.g., file paths, chemical names) that is NOT presented within a documentary context (like a README table).
3.  **Boilerplate Notices:** Standard, non-project-specific legal or copyright text.

**Overriding Principle:** The functional category of the content is the most important factor. For example, a well-written software license is still boilerplate and must be eliminated. A well-structured README with code snippets is still a guide and must be kept.

### Task-Specific Guiding Principle for Plausible Reasoning
For the s0 task, reasoning is considered **plausible** if it correctly identifies the category of the text based on the Ground Truth above.
- **Plausible reasoning for KEEPING:** "This is human documentation," "This is an instructional guide," "This is API documentation," "This is a bug report."
- **Plausible reasoning for ELIMINATING:** "This is a log file," "This is a stack trace," "This is a boilerplate license."
A simple, correct categorization is sufficient.

### VERIFICATION SCRIPT & RESPONSE FORMAT

You **must** respond with a single, raw JSON object. Fill out the fields sequentially as you perform the verification.

**Step 1: Identify the Core Rule**
   - Read the `<original_system_prompt>` and the `<original_prompt>`. The complete instructions for the original AI are contained within these two tags.
   - **Search both prompts for the main instructions that define the AI's classification task (e.g., look for sections like "Instructions", "Keep Content That", or "Eliminate Content That").**
   - **You MUST ignore any final meta-instructions about formatting or any "Now analyze..." command found in the `<original_prompt>`.**
   - Quote the single most important sentence that defines the primary classification rule. This is your ground truth.
   - Populate `analysis_core_rule`.

**Step 2: Perform a Two-Point Comparison Checklist**
   - **Check 1: Decision Correctness.** Read the `<source_data>` and the main decision in `<ai_output_to_verify>`. Is the AI's main decision a correct application of the `analysis_core_rule` to the source data? Answer "Yes" or "No". Populate `analysis_is_decision_correct`.
   - **Check 2: Reasoning Plausibility.** Read the reasoning in `<ai_output_to_verify>`. According to the **Task-Specific Guiding Principle** above, is this a plausible justification for the decision? Answer "Yes" or "No". Populate `analysis_is_reasoning_plausible`.

**Step 3: Determine Final Verdict**
   - Strictly apply the following logic tree based on your answers in Step 2.
   - **IF `analysis_is_decision_correct` is "No"**: The `evaluation` **MUST** be **`incorrect`**.
   - **IF `analysis_is_decision_correct` is "Yes"` AND `analysis_is_reasoning_plausible` is "No"**: The `evaluation` **MUST** be **`partially correct`**.
   - **IF `analysis_is_decision_correct` is "Yes"` AND `analysis_is_reasoning_plausible` is "Yes"**: The `evaluation` **MUST** be **`correct`**.
   - Populate the `evaluation` field. Then, write a one-sentence final `reasoning` that states your verdict and confirms the status of the decision and reasoning.

```json
{{
  "analysis_core_rule": "The core rule was to...",
  "analysis_is_decision_correct": "Yes" | "No",
  "analysis_is_reasoning_plausible": "Yes" | "No",
  "evaluation": "correct" | "partially correct" | "incorrect",
  "reasoning": "My verdict is [evaluation] because the main decision was [correct/incorrect] based on the core rule, and the reasoning was [plausible/implausible]."
}}
```
"""

def main():
    NoiseFilteringStageVerification(hostname=LLMHost.GREEN_LAB, batch_size_override=20, disable_cache=True).execute_verification()


if __name__ == "__main__":
    main()
