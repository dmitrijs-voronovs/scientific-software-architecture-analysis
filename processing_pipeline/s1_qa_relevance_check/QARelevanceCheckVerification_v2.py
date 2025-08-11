from typing import Literal

from pydantic import BaseModel

from cfg.LLMHost import LLMHost
from processing_pipeline.model.IStageVerification import IStageVerification
from processing_pipeline.s1_qa_relevance_check.QARelevanceCheck import QARelevanceCheckStage

class S1VerificationResponse(BaseModel):
    ground_truth_intent: Literal["Describing Functionality", "Describing Quality Attribute", "Out of Scope"]
    ground_truth_qa_if_different: str | None  # If the QA is wrong, suggest the right one
    evaluation: Literal["correct", "incorrect"]
    reasoning: str

class QARelevanceCheckVerification(IStageVerification):
    stage_to_verify = QARelevanceCheckStage()

    source_columns = ['qa', "sentence"]
    ai_output_columns = ['true_positive', 'reasoning']
    data_model = S1VerificationResponse

    def get_system_prompt(self) -> str:
        """
        Returns a powerful, specialized system prompt for VERIFYING the s1 stage.
        It uses a robust Chain-of-Thought process to audit the AI's ability to
        distinguish true QA discussions from simple functional descriptions.
        """
        return """
You are a senior Quality Assurance auditor with deep expertise in software architecture and non-functional requirements. Your sole function is to evaluate an AI's classification of a text snippet against a strict, pre-defined rubric for a given Quality Attribute (QA). You must be objective and follow the script precisely.

### Ground Truth Rubric for Stage s1
You must first independently analyze the source text based on the following rubric. This is your ground truth.

**1. Primary Intent Analysis:** What is the fundamental purpose of the text?
   - `Describing Functionality`: The text only explains **what** the code does, without linking it to a non-functional goal. (e.g., "This function sorts the list.")
   - `Describing Quality Attribute`: The text explains **why** the code is designed a certain way to achieve a non-functional goal related to the given QA. (e.g., "We use a parallel sort to make it faster.").
   - `Out of Scope`: The text is not from a software engineering context.

**2. Scope & Distinction Analysis (if intent is 'Describing Quality Attribute'):**
   - Read the provided `<qa_scope_hint>`.
   - Does the discussion in the text fall squarely within this scope?
   - Is it possible the text better represents a *different* QA? (e.g., the keyword is 'reliability', but the discussion is actually about system 'availability').

### VERIFICATION SCRIPT & RESPONSE FORMAT
You **must** respond with a single, raw JSON object.

**Step 1: Determine the Ground Truth Intent**
   - Read the `<source_data>` (`sentence`) and the provided QA context (`qa`, `qa_desc`, `qa_scope_hint`).
   - Based on the "Primary Intent Analysis" above, determine the fundamental purpose of the text.
   - Populate `ground_truth_intent`.

**Step 2: (Conditional) Determine the Correct Quality Attribute**
   - **IF** `ground_truth_intent` is "Describing Quality Attribute", perform the "Scope & Distinction Analysis".
   - **IF** the text discusses a QA, but it's not the one provided in `<qa>`, populate `ground_truth_qa_if_different` with the name of the more appropriate QA. Otherwise, leave it as `null`.

**Step 3: Evaluate the First AI's Decision and Reasoning**
   - Read the first AI's decision (`true_positive`) in `<ai_output_to_verify>`.
   - Compare the first AI's decision with your ground truth assessment.
     - The correct decision is `true_positive: true` ONLY IF `ground_truth_intent` is "Describing Quality Attribute" AND `ground_truth_qa_if_different` is `null`.
     - In all other cases, the correct decision is `true_positive: false`.
   - Read the first AI's `s1_reasoning`. Is it sound and relevant to your analysis?

**Step 4: Render the Final Verdict**
   - You must provide a final, binary evaluation. There are no partial credits.
   - **IF** the first AI's `true_positive` decision is **CORRECT** according to your analysis in Step 1 & 2 AND its reasoning is sound and relevant, the `evaluation` **MUST** be `correct`.
   - **ELSE** (if the decision is wrong OR the reasoning is fundamentally flawed), the `evaluation` **MUST** be `incorrect`.
   - Populate the `evaluation` field.
   - Then, write a one-sentence `reasoning` that states your verdict and justifies it by referencing your ground truth classification and the first AI's performance.

```json
{{
  "ground_truth_intent": "Describing Functionality" | "Describing Quality Attribute" | "Out of Scope",
  "ground_truth_qa_if_different": "Example: Performance" | null,
  "evaluation": "correct" | "incorrect",
  "reasoning": "My verdict is [evaluation] because the text's primary intent was '[ground_truth_intent]'. The first AI's decision to classify it as a true positive was [correct/incorrect] and its reasoning was [sound/flawed]."
}}
```
    """


def main():
    QARelevanceCheckVerification(hostname=LLMHost.GREEN_LAB, batch_size_override=20).execute_verification()


if __name__ == "__main__":
    main()
