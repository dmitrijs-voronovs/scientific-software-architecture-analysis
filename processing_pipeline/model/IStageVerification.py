from abc import ABC, abstractproperty, abstractmethod
from pydoc import html
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from cfg.ModelName import ModelName
from constants.abs_paths import AbsDirPath
from processing_pipeline.model.CSVDFHandler import CSVDFHandler
from processing_pipeline.model.IBaseStage import IBaseStage


class OllamaFormatValidityResponse(BaseModel):
    # analysis_source_summary: "str"
    # analysis_decision_summary: "str"
    # analysis_reasoning_summary: "str"
    analysis_core_rule: str
    analysis_is_decision_correct: Literal["Yes", "No"]
    analysis_is_reasoning_plausible: Literal["Yes", "No"]
    evaluation: Literal["correct", "partially correct", "incorrect"]
    reasoning: str


class IStageVerification(IBaseStage, ABC):
    temperature = 0.0
    model_name = ModelName.DEEPSEEK_8B
    in_dir = AbsDirPath.SAMPLES
    out_dir = AbsDirPath.SAMPLES_VERIFIED
    cache_dir = AbsDirPath.CACHE / "samples"
    DFHandler = CSVDFHandler()
    data_model = OllamaFormatValidityResponse

    stage_to_verify: IBaseStage

    @property
    def stage_prefix(self) -> str:
        return self.stage_to_verify.stage_name

    @property
    @abstractmethod
    def source_columns(self) -> list[str]:
        """A list of column names that contain the source text(s) for the AI."""
        pass

    @property
    def prompt_column(self) -> str:
        return self.stage_prefix + "_prompt"

    @property
    @abstractmethod
    def ai_output_columns(self) -> list[str]:
        """
        A list of the AI output column SUFFIXES to be verified.
        (e.g., ['to_eliminate', 'reasoning'] for stage 's0')

        """
        pass

    def get_system_prompt(self) -> str:
        """
        Returns a TRULY GENERIC system prompt for the base verifier.
        It contains NO task-specific examples. Stage-specific verifiers
        will inherit and override this method.
        """
        return """
You are a Quality Assurance bot. Your only function is to execute a structured verification script and produce a JSON output. You must be objective and strictly follow the checklist below. Do not introduce outside criteria or opinions.

### Guiding Principle for Evaluating Reasoning
Your evaluation **MUST** be pragmatic. You are not a literary critic. The first AI's output does not need to be perfect. The reasoning is considered **plausible** if it is a **brief, relevant justification** for the main decision based on the rules it was given.

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

    def to_prompt(self, x: pd.Series) -> str:
        """
        Generates the DYNAMIC user prompt.
        It contains only the specific data for this one evaluation item, formatted as defined
        in the system prompt.
        """
        # 1. Prepare the source text block
        source_text_lines = [f"<{col}>{html.escape(str(x.get(col, 'N/A')))}</{col}>" for col in self.source_columns]
        source_text_str = "\n".join(source_text_lines)

        # 2. Prepare the original prompt string
        original_prompt_str = html.escape(x.get(self.prompt_column, 'N/A'))

        # 3. Prepare the AI output block
        ai_output_lines = []
        for col_suffix in self.ai_output_columns:
            full_col_name = f"{self.stage_prefix}_{col_suffix}"
            value = str(x.get(full_col_name, 'N/A'))
            ai_output_lines.append(f"    <{col_suffix}>{html.escape(value)}</{col_suffix}>")
        ai_output_block_str = "\n".join(ai_output_lines)

        # 4. Assemble the final data block for the user message
        return f"""Now, perform your your audit based on the data below.

<evaluation_data>
    <original_system_prompt>
    {self.stage_to_verify.get_system_prompt()}
    </original_system_prompt>
    <original_prompt>
    {original_prompt_str}
    </original_prompt>

    <source_data>
    {source_text_str}
    </source_data>

    <ai_output_to_verify>
    {ai_output_block_str}
    </ai_output_to_verify>
</evaluation_data>
"""

    @property
    def stage_name(self) -> str:
        return self.stage_to_verify.stage_name + '_v'

    def execute_verification(self):
        self.execute([self.stage_to_verify.stage_name])
