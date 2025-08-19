from typing import Literal, Optional

from pydantic import BaseModel

from cfg.LLMHost import LLMHost
from processing_pipeline.model.IStageVerification import IStageVerification
from processing_pipeline.s1_qa_relevance_check.QARelevanceCheck_v2 import QARelevanceCheckStage_v2


class S1VerificationResponseV6(BaseModel):
    """
    Defines the structured output for the S1 verifier, focusing on auditing
    the executor's reasoning for validity and defensibility.
    """
    is_executor_reasoning_valid: bool
    evaluation: Literal["correct", "incorrect"]
    reasoning: str


class QARelevanceCheckVerification_v2(IStageVerification):
    """
    This class implements the verification logic for the QARelevanceCheckStage_v2.
    It uses a new "auditor" prompt that focuses on validating the executor's
    reasoning rather than forming an independent opinion.
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
    data_model = S1VerificationResponseV6

    def get_system_prompt(self) -> str:
        """
        Returns the system prompt for the verifier LLM. This prompt reframes the
        task from independent analysis to direct auditing of the executor's logic.
        """
        return """
### Persona ###
You are a Quality Assurance auditor. Your job is not to have an opinion, but to verify if a junior AI architect's work is logical and follows the provided instructions. You are receptive and trust the junior AI unless it makes a clear and obvious mistake.

### Core Task: Audit, Don't Re-analyze ###
Your goal is to evaluate if the AI architect's JSON output (`ai_output_to_verify`) is a **reasonable and defensible conclusion** based on the `source_data` and the rules in the `<original_prompt>`.

### The "Trust but Verify" Principle ###
Your default stance is that the junior AI is correct. Your job is to find clear evidence to the contrary.

1.  **Start with the Executor's Conclusion:** Look at the executor's `analysis_rubric_check` and `reasoning` fields first.
2.  **Check for Validity:** Read the `sentence` and the rubric from the `<original_prompt>`. Is the executor's conclusion plausible?
    * If the rubric says "Mentions of package managers (pip...)" is an Inclusion Criterion, and the executor flags a sentence with "pip install" as `True`, then its reasoning is **VALID**.
    * If the rubric says "User Installation/Configuration Errors" is an Exclusion Criterion, and the executor flags an "OSError: file not found" message as `False`, its reasoning is **VALID**.
    * If the executor's reasoning directly contradicts a clear rule or the content of the sentence, it is **INVALID**.
3.  **Your Goal is Receptiveness:** Do not fail the executor for minor differences in wording. If its logic is sound and leads to the correct `true_positive` result, you must consider its reasoning valid.

### Verification & Audit Process ###

1.  **Assess Executor's Reasoning:**
    * Read the executor's entire output in `<ai_output_to_verify>`.
    * Read the `source_data` and the rubric from the `<original_prompt>`.
    * Based on the "Trust but Verify" principle, is the executor's chain of thought and final conclusion valid and defensible?
    * `is_executor_reasoning_valid`: Set to `True` if the reasoning is plausible and follows the rules; otherwise, set to `False`.

2.  **Render Final Verdict:**
    * `evaluation`: This is a direct reflection of your audit. If `is_executor_reasoning_valid` is `True`, the `evaluation` MUST be `correct`. If it is `False`, the `evaluation` MUST be `incorrect`.
    * `reasoning`: Write a concise, one-sentence justification for your verdict. Start by stating if the executor's reasoning was valid and why. (e.g., "The executor's reasoning was valid because it correctly identified that the text matched the 'pip' inclusion criterion." or "The executor's reasoning was invalid because it misidentified a clear 'Problem' as a 'Solution'.")

### Mandatory Output Format ###
You MUST provide your response as a single JSON object.
```json
{
  "is_executor_reasoning_valid": "boolean",
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
