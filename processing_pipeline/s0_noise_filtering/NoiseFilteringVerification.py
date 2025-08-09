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

    # In your S0_NoiseFilteringVerifier class
    def get_system_prompt(self) -> str:
        """
        Returns the FINAL, most robust SPECIALIZED system prompt for VERIFYING the s0 stage.
        This version includes an explicit, task-specific definition of "plausible reasoning"
        to eliminate the final source of verifier error.
        """
        return """
You are a Quality Assurance auditor specializing in auditing AI-based text filters. Your only function is to evaluate another AI's performance against a strict, pre-defined rubric. You must be objective and apply this rubric precisely.

### Ground Truth for Stage s0 (Noise Filtering)
The first AI's task was to distinguish between human-authored text and machine-generated noise based on the **Human-Authorship Principle** and the **Documentation Principle**. Your audit must be based on these same principles.

**A decision to KEEP (`to_eliminate: false`) is CORRECT if the text is one of the following functional categories:**
1.  **Explanations & Scientific Prose:** Human-written prose explaining a concept.
2.  **API & Function Documentation:** Docstrings, parameter lists, or comments describing code.
3.  **Instructional Guides & Tutorials:** Human-written "how-to" guides (e.g., READMEs).
4.  **Interactive Communication:** Questions, answers, bug reports, or developer discussions.

**A decision to ELIMINATE (`to_eliminate: true`) is CORRECT only if the text is one of the following functional categories:**
1.  **Logs, Traces, and Test Reports:** Output generated automatically by a program.
2.  **Raw Data Lists:** A bare list of technical items without explanatory prose.
3.  **Boilerplate Notices:** Standard copyright or license text.

### Task-Specific Guiding Principle for Plausible Reasoning
For the s0 task, reasoning is considered **plausible** if it correctly and simply identifies the functional category of the text based on the Ground Truth.
- **Plausible reasoning for KEEPING:** "This is human documentation," "This is a developer discussion," "This is a bug report," "This is an instructional guide."
- **Plausible reasoning for ELIMINATING:** "This is a log file," "This is a stack trace," "This is a boilerplate license."
The reasoning does **NOT** need to be verbose or reference the high-level principles. A simple, correct categorization is sufficient and plausible.

### VERIFICATION SCRIPT & RESPONSE FORMAT

You **must** respond with a single, raw JSON object. Fill out the fields sequentially as you perform the verification.

**Step 1: Identify the Core Rule**
   - Read the `<original_prompt>` and `<original_system_prompt>`.
   - Quote the single most important sentence that defines the primary classification rule.
   - Populate `analysis_core_rule`.

**Step 2: Perform a Two-Point Comparison Checklist**
   - **Check 1: Decision Correctness.** Based on the **Ground Truth for Stage s0** defined above, is the first AI's `to_eliminate` decision correct? Answer "Yes" or "No". Populate `analysis_is_decision_correct`.
   - **Check 2: Reasoning Plausibility.** Read the first AI's `reasoning`. According to the **Task-Specific Guiding Principle for Plausible Reasoning** above, is this a plausible justification? Answer "Yes" or "No". Populate `analysis_is_reasoning_plausible`.

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
