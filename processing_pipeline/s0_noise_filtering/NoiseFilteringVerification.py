import html
from pathlib import Path

import pandas as pd

from cfg.LLMHost import LLMHost
from constants.abs_paths import AbsDirPath
from processing_pipeline.model.IStageVerification import IStageVerification
from processing_pipeline.s0_noise_filtering.NoiseFiltering import NoiseFilteringStage


class NoiseFilteringStageVerification(IStageVerification):
    stage_to_verify = NoiseFilteringStage()

    source_columns = ['sentence']
    ai_output_columns = ['to_eliminate', 'reasoning']

    # In your IStageVerification class for s0:

    def get_system_prompt(self) -> str:
        """
        Returns a specialized system prompt for VERIFYING the s0 (Noise Filtering) stage.
        This prompt contains the specific ground truth and nuances for this stage.
        """
        return """
You are a meticulous Quality Assurance auditor. Your sole function is to evaluate the performance of another AI (the "first AI") on a text classification task. You must be objective and strictly follow the verification script below.

### Ground Truth for Stage s0 (Noise Filtering)
The first AI's task was to act as a broad-pass filter. Its goal was to **eliminate only unambiguous, non-prose artifacts** while keeping anything with a meaningful, human-written natural language component.

**Correct "to_eliminate: false" (KEEP) Decision:**
The first AI was **CORRECT** to keep the content if it was any of the following:
1.  **Interactive Communication:** Bug reports, critiques, suggestions, or discussions between developers.
2.  **Detailed Documentation:** Prose that explains implementation strategies, trade-offs, or usage (e.g., explaining different API strategies).
3.  **Simple Explanations:** Basic code comments or docstrings that describe, in natural language, what a function or component does (e.g., "Converts a UTF8 sequence to UTF32.").

**Correct "to_eliminate: true" (ELIMINATE) Decision:**
The first AI was **CORRECT** to eliminate the content only if it was one of the following:
1.  **Machine-Generated Output:** Program logs, error traces, or build logs.
2.  **Lists of Artifacts:** A list of API functions, file paths, or variables without any explanatory prose.
3.  **Low-Value Metadata:** Simple changelogs (e.g., "Version 1.5, fix bug") that lack a narrative.
4.  **Pure Code/Syntax:** Executable code with no explanatory comments.

**Crucial Distinction:** A simple docstring like "Returns the default graphics context in use" should be **KEPT** by the first AI. A list of API names like "scanpy.neighbors.connectivities" should be **ELIMINATED**. You must apply this distinction when judging the first AI's decision.

### VERIFICATION SCRIPT & RESPONSE FORMAT
You **must** respond with a single, raw JSON object.

**Step 1: Identify the Core Rule**
   - Read the `<original_system_prompt>` and `<original_prompt>`.
   - Quote the single most important sentence that defines the original AI's primary classification rule. This is what the *first AI* was told to do.
   - Populate `analysis_core_rule`.

**Step 2: Perform a Two-Point Comparison Checklist**
   - **Check 1: Decision Correctness.** Read the `<source_data>` and the `<ai_output_to_verify>`. According to the **Ground Truth for Stage s0** defined above, is the first AI's `to_eliminate` decision correct? Answer "yes" or "no". Populate `analysis_is_decision_correct`.
   - **Check 2: Reasoning Plausibility.** Read the first AI's `reasoning`. Is it a brief, relevant justification for its decision, even if the decision was wrong? (e.g., if it eliminated a docstring because it was a "technical artifact," the reasoning is plausible for that decision). Answer "yes" or "no". Populate `analysis_is_reasoning_plausible`.

**Step 3: Determine Final Verdict**
   - Strictly apply the following logic tree based on your answers in Step 2.
   - **IF `analysis_is_decision_correct` is "no"**: The `evaluation` **MUST** be **`incorrect`**.
   - **IF `analysis_is_decision_correct` is "yes"` AND `analysis_is_reasoning_plausible` is "no"**: The `evaluation` **MUST** be **`partially correct`**.
   - **IF `analysis_is_decision_correct` is "yes"` AND `analysis_is_reasoning_plausible` is "yes"**: The `evaluation` **MUST** be **`correct`**.
   - Populate `evaluation` and write a one-sentence `reasoning` that states your verdict and confirms why.

```json
{{
  "analysis_core_rule": "The core rule from the original prompt was to...",
  "analysis_is_decision_correct": "yes" | "no",
  "analysis_is_reasoning_plausible": "yes" | "no",
  "evaluation": "correct" | "partially correct" | "incorrect",
  "reasoning": "My verdict is [evaluation] because the decision was [correct/incorrect] based on the ground truth, and the reasoning was [plausible/implausible]."
}}
```
    """


def main():
    NoiseFilteringStageVerification(hostname=LLMHost.GREEN_LAB, batch_size_override=20, disable_cache=True).execute_verification()


if __name__ == "__main__":
    main()
