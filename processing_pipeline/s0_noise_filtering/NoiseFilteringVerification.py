import html
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from cfg.LLMHost import LLMHost
from processing_pipeline.model.IStageVerification import IStageVerification
from processing_pipeline.s0_noise_filtering.NoiseFiltering import NoiseFilteringStage


class OllamaFormatValidityResponse(BaseModel):
    correctness: Literal["correct", "partially correct", "incorrect"]
    reasoning: str


class NoiseFilteringStageVerification(IStageVerification):
    data_model = OllamaFormatValidityResponse
    stage_to_verify = NoiseFilteringStage

    @classmethod
    def to_prompt(cls, x: pd.Series) -> str:
        original_prompt_str = html.escape(x.get('s0_prompt', 'N/A'))
        to_eliminate_str = str(x.get('s0_to_eliminate', 'N/A')).lower()
        reasoning_str = html.escape(x.get('s0_reasoning', 'N/A'))

        return f"""
You are a meticulous and expert evaluator of AI model outputs. Your goal is to review and verify a classification made by another AI model.

You will be provided with the original prompt that the first AI was given, and the output it produced. These will be enclosed in XML tags. Your task is to judge if the AI's output was `correct`, `partially correct`, or `incorrect` based on the instructions contained within the `<original_prompt>` tag.

---
## Data for Evaluation

<evaluation_data>
    <original_prompt>
    {original_prompt_str}
    </original_prompt>

    <ai_output_to_verify>
        <decision_to_eliminate>{to_eliminate_str}</decision_to_eliminate>
        <reasoning>{reasoning_str}</reasoning>
    </ai_output_to_verify>
</evaluation_data>

---
## Your Evaluation Task

1.  **Analyze the Task:** First, carefully read the instructions inside the `<original_prompt>` tag to understand the task the first AI was supposed to perform.
2.  **Evaluate the Output:** Next, examine the `<ai_output_to_verify>`. Judge whether the decision and reasoning are consistent with the instructions from the `<original_prompt>`.
3.  **Provide Your Verdict:** Based on your analysis, provide your evaluation using the strict, mutually exclusive definitions below.

### Evaluation Criteria

- **`correct`**:
  - The `decision_to_eliminate` is **correct** according to the rules in `<original_prompt>`.
  - AND the `reasoning` accurately and logically justifies that decision.

- **`partially correct`**:
  - The `decision_to_eliminate` is **correct**.
  - BUT the `reasoning` is flawed. This includes reasoning that is weak, imprecise, irrelevant, missing when required, or uses the wrong justification.

- **`incorrect`**:
  - The `decision_to_eliminate` is **incorrect**.
  - This verdict applies **regardless of the quality of the `reasoning`**. If the main decision is wrong, the entire output is incorrect. This explicitly includes scenarios where the `reasoning` accurately describes the content but leads to the wrong final decision.

### Response Format
You **must** respond with a single, raw JSON object. Do not add any text, comments, or markdown formatting before or after the JSON. Your response must conform to the following structure:

```json
{{
    "correct": "correct" | "partially correct" | "incorrect",
    "reasoning": "Your detailed explanation for why you chose this evaluation. Justify your verdict by referencing the specific rules from the `<original_prompt>` and the content of `<ai_output_to_verify>`."
}}
"""


def main():
    NoiseFilteringStageVerification(hostname=LLMHost.GREEN_LAB, batch_size_override=10).execute_verification()


if __name__ == "__main__":
    main()
