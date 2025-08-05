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
    correctness: Literal["correct", "partially correct", "incorrect"]
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

    # In your BaseStageVerification class

    def get_system_prompt(self) -> str:
        """
        Returns the new PRAGMATIC generic system prompt.
        It forces a structured analysis but lowers the bar for a 'correct' verdict,
        focusing on the primary decision.
        """
        return """
You are a pragmatic and experienced Quality Assurance Lead. Your goal is to efficiently determine if an AI's output is functionally correct. You must prioritize the main decision and avoid nitpicking minor flaws in the reasoning if the conclusion is sound.

### VERIFICATION SCRIPT

You will receive an `<evaluation_data>` block. You must perform the following analysis to populate your JSON response.

**1. Summarize the Goal:**
   - Read the `<original_prompt>`.
   - In one sentence, what was the AI's primary objective?
   - Populate the `goal_summary` field.

**2. Summarize the Source Data:**
   - Read the `<source_data>`.
   - In one sentence, what is the nature of this data (e.g., documentation, code, error log, user question)?
   - Populate the `source_summary` field.

**3. Summarize the AI's Output:**
   - Read the `<ai_output_to_verify>`.
   - What was the AI's main decision and its core justification?
   - Populate the `output_summary` field.

**4. Synthesize and Decide:**
   - **Guiding Principle:** Your primary concern is the correctness of the main decision. The reasoning only needs to be a plausible justification, not a perfect or exhaustive analysis.
   - Compare the AI's main decision against the goal and the source data.
   - Based on this principle, choose your `evaluation` verdict according to the criteria below.

### EVALUATION CRITERIA (Lowered Threshold)

- **`correct`**: The AI's main decision (e.g., the boolean flag, the primary classification) is **correct**. The `reasoning` is a **plausible and relevant justification**, even if it is not perfectly detailed or exhaustive. This is the default verdict if the AI understood the task and got the main point right.
- **`partially correct`**: The AI's main decision is **correct**, BUT the `reasoning` is **factually wrong, completely irrelevant, or nonsensical**. This is for cases where the AI got the right answer for the wrong reason (i.e., by accident).
- **`incorrect`**: The AI's main decision is **fundamentally wrong**.

### RESPONSE FORMAT

You **must** respond with a single, raw JSON object. First, fill in the `analysis` object. Then, use that analysis to determine your final `evaluation` and `reasoning`.

```json
{{
  "analysis": {{
    "goal_summary": "The AI's primary objective was to...",
    "source_summary": "The source data is...",
    "output_summary": "The AI decided that... because..."
  }},
  "evaluation": "correct" | "partially correct" | "incorrect",
  "reasoning": "My verdict is [evaluation]. The AI's main decision was [correct/incorrect]. The reasoning provided was [plausible/flawed/irrelevant], leading to the final verdict."
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
