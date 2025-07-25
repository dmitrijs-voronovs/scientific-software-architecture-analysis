import html

import pandas as pd

from cfg.LLMHost import LLMHost
from processing_pipeline.model.IStageVerification import IStageVerification
from processing_pipeline.s1_qa_relevance_check.QARelevanceCheck import QARelevanceCheckStage


class QARelevanceCheckVerification(IStageVerification):
    stage_to_verify = QARelevanceCheckStage

    @classmethod
    def to_prompt(cls, x: pd.Series) -> str:
        original_prompt_str = html.escape(x.get('s1_prompt', 'N/A'))
        is_true_positive_str = str(x.get('s1_true_positive', 'N/A')).lower()
        reasoning_str = html.escape(x.get('s1_reasoning', 'N/A'))

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
        <is_true_positive>{is_true_positive_str}</is_true_positive>
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
  - The decision in `<is_true_positive>` is **correct** according to the rules in `<original_prompt>`.
  - AND the `reasoning` accurately and logically justifies that decision.

- **`partially correct`**:
  - The decision in `<is_true_positive>` is **correct**.
  - BUT the `reasoning` is flawed. This includes reasoning that is weak, imprecise, irrelevant, missing when required, or uses the wrong justification.

- **`incorrect`**:
  - The decision in `<is_true_positive>` is **incorrect**.
  - This verdict applies **regardless of the quality of the `reasoning`**. If the main decision is wrong, the entire output is incorrect.

### Response Format
You **must** respond with a single, raw JSON object. Do not add any text, comments, or markdown formatting before or after the JSON. Your response must conform to the following structure:

```json
{{
    "correct": "correct" | "partially correct" | "incorrect",
    "reasoning": "Your detailed explanation for why you chose this evaluation. Justify your verdict by referencing the specific rules from the `<original_prompt>` and the content of `<ai_output_to_verify>`."
}}

Now, perform your evaluation based on the content within the <evaluation_data> block.
"""


def main():
    QARelevanceCheckVerification(hostname=LLMHost.RADU_SERVER, batch_size_override=20).execute_verification()


if __name__ == "__main__":
    main()
