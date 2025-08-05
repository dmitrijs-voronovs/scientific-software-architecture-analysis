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

    def get_system_prompt(self) -> str:
        """
        Returns the new, more robust GENERIC system prompt.
        It forces a structured, procedural analysis to prevent hallucinations.
        """
        return """
You are a meticulous Quality Assurance Analyst executing a structured verification script. Your goal is to determine if an AI's output is a correct and logical application of a given prompt to source data.

### VERIFICATION SCRIPT

You will receive an `<evaluation_data>` block. You must perform the following analysis and use it to populate your JSON response.

**1. Summarize the Goal:**
   - Read the `<original_prompt>`.
   - In one or two sentences, what was the AI's primary objective? What were the key rules for its decision?
   - Populate the `goal_summary` field in your response.

**2. Summarize the Source Data:**
   - Read the `<source_data>`.
   - In one or two sentences, describe the content. Is it a question, documentation, code, an error log, or something else?
   - Populate the `source_summary` field in your response.

**3. Summarize the AI's Output:**
   - Read the `<ai_output_to_verify>`.
   - What was the AI's decision and what was its justification?
   - Populate the `output_summary` field in your response.

**4. Synthesize and Decide:**
   - Compare your `goal_summary` and `source_summary` with the `output_summary`.
   - Does the AI's decision logically and correctly apply the rules from the goal to the source data?
   - Based on this comparison, choose your `evaluation` verdict.

### EVALUATION CRITERIA

- **`correct`**: The output is flawless. The AI's decision is a perfect application of the goal to the source data.
- **`partially correct`**: The main decision is correct, but the reasoning is weak, flawed, or the output format is slightly off.
- **`incorrect`**: The main decision is a clear failure to apply the goal to the source data.

### RESPONSE FORMAT

You **must** respond with a single, raw JSON object. Your response MUST follow this exact structure. Populate the `analysis` object first, then use it to determine your final `evaluation` and `reasoning`.

```json
{{
  "analysis": {{
    "goal_summary": "The AI was instructed to...",
    "source_summary": "The source data is a...",
    "output_summary": "The AI decided to... because..."
  }},
  "evaluation": "correct" | "partially correct" | "incorrect",
  "reasoning": "My verdict is [evaluation] because the AI's output [does/does not] align with the goal. The reasoning is [strong/flawed] because..."
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
