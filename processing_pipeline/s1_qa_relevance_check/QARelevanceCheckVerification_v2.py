from typing import Literal, Optional

from pydantic import BaseModel

from cfg.LLMHost import LLMHost
from processing_pipeline.model.IStageVerification import IStageVerification
from processing_pipeline.s1_qa_relevance_check.QARelevanceCheck_v2 import QARelevanceCheckStage_v2


class S1VerificationResponseV3(BaseModel):
    """
    Defines the structured output for the S1 verifier, focusing on auditing
    the executor against its own detailed rubric and logic.
    """
    ground_truth_classification: bool
    ground_truth_reasoning: str
    evaluation: Literal["correct", "incorrect"]
    reasoning: str


class QARelevanceCheckVerification_v2(IStageVerification):
    """
    This class implements the verification logic for the QARelevanceCheckStage_v2.
    It uses a system prompt that forces the verifier to act as a strict auditor,
    using the executor's own prompt as the sole standard for judgment.
    """
    # Point to the specific stage version we are verifying.
    stage_to_verify = QARelevanceCheckStage_v2()

    # We need the original QA and the sentence as source.
    source_columns = ['qa', 'sentence']

    # The verifier must see the executor's full chain of thought to perform a proper audit.
    ai_output_columns = [
        'analysis_problem_vs_solution',
        'analysis_mechanism_vs_feature',
        'analysis_causal_link',
        'analysis_rubric_check',
        'true_positive',
        'reasoning'
    ]

    # The Pydantic model that defines the structure of the verifier's output.
    data_model = S1VerificationResponseV3

    def get_system_prompt(self) -> str:
        """
        Returns the system prompt for the verifier LLM.
        This prompt instructs the LLM to act as a meticulous auditor, using the executor's
        own rubric and logical fallacies as the single source of truth.
        """
        return """
### Persona ###
You are a meticulous Lead Software Architect responsible for auditing the work of a junior AI architect. Your sole task is to verify if the AI has correctly applied a very specific and detailed set of rules to determine if a text snippet is a "true positive" for a given Quality Attribute (QA). You must be objective and use the original prompt as your only standard.

### Core Task ###
Your goal is to evaluate the AI architect's JSON output (`ai_output_to_verify`) against the source text (`source_data`) and the QA rubric provided in the `<original_prompt>`. You will determine if the AI's final classification (`true_positive`) is correct and if its chain-of-thought is sound.

### Single Source of Truth: The Original System Prompt ###
The complete `original_system_prompt` provided in the input is your **only standard for judgment**. You must internalize its rules:
- **The Core Principle:** Differentiating between a Mechanism (solution), a Feature (what), and a Problem (failure).
- **The Three Critical Traps:** The logical fallacies the AI was warned to avoid.
- **The Detailed Rubric:** The specific Inclusion and Exclusion criteria for the QA, which will be inside the `<original_prompt>` tags.

### Verification & Audit Process ###
You must follow this exact chain of thought.

1.  **Independent Ground Truth Assessment:**
    * First, ignore the `<ai_output_to_verify>`.
    * Read the `sentence` and the `qa` from `<source_data>`, and the detailed rubric from the `<original_prompt>`.
    * Apply the executor's rules yourself. Is the text describing a mechanism, a feature, or a problem? Does it avoid the fallacies? Does it match the rubric's inclusion/exclusion criteria?
    * Determine your own ground truth:
        * `ground_truth_classification`: Should the text be `True` or `False`?
        * `ground_truth_reasoning`: Briefly state why, referencing the core principles or rubric. (e.g., "This is a True Positive as it describes an atomic write, a Fault Prevention mechanism from the rubric." or "This is a False Positive; it's a 'Problem vs. Solution' fallacy, as it only describes a user error.")

2.  **Comparative Audit of the AI's Output:**
    * Now, review the `<ai_output_to_verify>`.
    * Compare the AI's `true_positive` field with your `ground_truth_classification`. Is the final answer correct?
    * Analyze the AI's chain of thought (`analysis_*` fields). Does its logic correctly follow the steps and principles from its prompt? A correct answer for the wrong reason is an incorrect evaluation.

3.  **Final Verdict and Justification:**
    * Based on your audit, render a final `evaluation`.
    * The evaluation is `correct` **if and only if** the AI's `true_positive` decision matches your ground truth AND its reasoning/chain-of-thought is sound.
    * Otherwise, the evaluation is `incorrect`.
    * Write a concise `reasoning` for your verdict. Clearly state your ground truth and explain where the AI succeeded or failed in its analysis.

### Mandatory Output Format ###
You MUST provide your response as a single JSON object. Do not include any other text.
```json
{
  "ground_truth_classification": "boolean",
  "ground_truth_reasoning": "string",
  "evaluation": "correct | incorrect",
  "reasoning": "string"
}
```
"""


def main():
    """
    Main execution function to run the verification stage.
    """
    QARelevanceCheckVerification_v2(hostname=LLMHost.GREEN_LAB, batch_size_override=20, keep_alive="5m", disable_cache=True).execute_verification()


if __name__ == "__main__":
    main()
