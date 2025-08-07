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
        Returns a SPECIALIZED system prompt for VERIFYING the s0 stage.
        This prompt contains the specific ground truth and nuances for the
        noise filtering task.
        """
        return """
You are a Quality Assurance auditor specializing in noise detection in text. Your task is to evaluate another AI's ability to distinguish between human-authored prose and machine-generated noise.

### Ground Truth for Stage s0 (Noise Filtering)
The first AI's task was to filter out unambiguous programmatic noise based on the **Human-Authorship Principle**.

**A decision to KEEP (`to_eliminate: false`) is CORRECT if the text is:**
1.  **Explanations & Rationale:** Prose explaining what something is, how it works, or why a decision was made. This includes detailed documentation AND simple one-sentence function descriptions.
2.  **Interactive Communication:** Questions, answers, bug reports, or developer discussions.
3.  **Documentation with Code:** Human-written prose that contains code snippets as examples. The prose is the primary signal.

**A decision to ELIMINATE (`to_eliminate: true`) is CORRECT only if the text is:**
1.  **Machine-Generated Output:** Raw logs, stack traces, compiler errors, or test suite failures.
2.  **Lists of Raw Data:** A bare list of technical items (e.g., file paths, API names) that is NOT part of a larger explanatory document.
3.  **Boilerplate Notices:** Standard copyright or license text.

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
   - **Check 2: Reasoning Plausibility.** Read the reasoning in `<ai_output_to_verify>`. According to the **Guiding Principle** above, is this a plausible justification? Answer "Yes" or "No". Populate `analysis_is_reasoning_plausible`.

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
