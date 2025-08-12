from typing import Literal

from pydantic import BaseModel

from cfg.LLMHost import LLMHost
from processing_pipeline.model.IStageVerification import IStageVerification
from processing_pipeline.s1_qa_relevance_check.QARelevanceCheck_v2 import QARelevanceCheckStage_v2


class S1VerificationResponse(BaseModel):
    ground_truth_intent: Literal["Describing Functionality", "Describing Quality Attribute", "Out of Scope"]
    ground_truth_qa_if_different: str | None  # If the QA is wrong, suggest the right one
    evaluation: Literal["correct", "incorrect"]
    reasoning: str


class QARelevanceCheckVerification_v2(IStageVerification):
    stage_to_verify = QARelevanceCheckStage_v2()

    source_columns = ['qa', "sentence"]
    ai_output_columns = [
        "analysis_context_check",
        "analysis_intent",
        "analysis_scope_match",
        "true_positive",
        "reasoning",
    ]
    data_model = S1VerificationResponse

    def get_system_prompt(self) -> str:
        return """
You are a senior Quality Assurance auditor with deep expertise in software architecture. Your function is to **audit** an AI's classification of a text snippet. You are the final authority. Your judgment must be objective, meticulous, and grounded in the provided rubric. No partial credit is given.

### Ground Truth Rubric (Your Internal Standard)

First, you will establish the ground truth by silently analyzing the source text against this rubric.

1.  **Primary Intent Analysis:** What is the fundamental purpose of the text?
    - `Describing Functionality`: Explains **what** the code does. (This is a negative case).
    - `Describing Quality Attribute`: Explains **why** the code is designed a certain way for a non-functional goal. (This is a potential positive case).
    - `Out of Scope`: Not from a software engineering context. (This is a negative case).

2.  **Scope & Distinction Analysis:** If the intent is `Describing Quality Attribute`, does it perfectly match the `<qa_scope_hint>`? Or does it better fit a different QA? A mismatch is a negative case.

### Verification Script & Response Format

You **must** respond with a single, raw JSON object following this mandated script.

**Step 1: Determine Your Ground Truth.**
- Read the `<source_data>` and QA context.
- Based on your private analysis using the rubric above, determine the `ground_truth_intent`.
- If the intent is 'Describing Quality Attribute' but for the *wrong* QA, populate `ground_truth_qa_if_different` with the correct QA name. Otherwise, it must be `null`.

**Step 2: Audit the First AI's Output.**
- Review the first AI's decision (`true_positive`) and its `reasoning` provided in `<ai_output_to_verify>`.
- Compare the AI's decision to your ground truth. The AI is correct ONLY IF `true_positive: true` matches a ground truth of `Describing Quality Attribute` with no QA mismatch.
- Critically evaluate the AI's reasoning. Is it logical, relevant, and free of contradictions? Flawed reasoning, even for a correct decision, constitutes a failure.

**Step 3: Render the Final Verdict.**
- `evaluation`: Set to `correct` **only if** the first AI's `true_positive` decision was correct AND its reasoning was sound. In all other cases—an incorrect decision OR flawed reasoning—you **must** set this to `incorrect`.
- `reasoning`: Write a concise, one-sentence justification for your verdict. Begin by stating your ground truth assessment and then comment on the correctness of the AI's decision and the quality of its reasoning.

```json
{{
  "ground_truth_intent": "Describing Functionality" | "Describing Quality Attribute" | "Out of Scope",
  "ground_truth_qa_if_different": "Example: Performance" | null,
  "evaluation": "correct" | "incorrect",
  "reasoning": "My verdict is [evaluation] because the text's primary intent was '[ground_truth_intent]'; the first AI's decision was [correct/incorrect] because its reasoning was [sound/flawed/contradictory]."
}}
```
"""


def main():
    QARelevanceCheckVerification_v2(hostname=LLMHost.GREEN_LAB, batch_size_override=20, disable_cache=True).execute_verification()


if __name__ == "__main__":
    main()
