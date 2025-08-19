from typing import Literal

from pydantic import BaseModel

from cfg.LLMHost import LLMHost
from processing_pipeline.model.IStageVerification import IStageVerification
from processing_pipeline.s3_tactic_extraction.TacticExtraction_v2 import TacticExtractionStage_v2


class S3VerificationResponseV2(BaseModel):
    """
    Defines the structured output for the S3 verifier, focusing on auditing
    the executor's reasoning for validity and defensibility.
    """
    is_executor_reasoning_valid: bool
    evaluation: Literal["correct", "incorrect"]
    reasoning: str

class TacticExtractionVerification(IStageVerification):
    stage_to_verify = TacticExtractionStage_v2()

    # Define the source text for the verifier
    source_columns = ['qa', 'sentence']

    # Define all the output columns from the S3 executor that the verifier needs to audit
    ai_output_columns = [
        'architectural_activity_extraction',
        'core_concept_analysis',
        'is_tactic_relevant',
        'relevance_reason',
        'tactic_evaluation',
        'selected_tactic',
        'justification'
    ]
    data_model = S3VerificationResponseV2

    def get_system_prompt(self) -> str:
        """
        Returns the system prompt for the verifier LLM. This prompt reframes the
        task from independent analysis to direct auditing of the executor's logic,
        promoting receptiveness to valid, if not perfect, answers.
        """
        return """
### Persona ###
You are a senior Quality Assurance auditor. Your job is not to have your own opinion, but to verify if a junior AI architect's work is logical, defensible, and follows its instructions. You are receptive and trust the junior AI unless it makes a clear and obvious mistake.

### Core Task: Audit, Don't Re-analyze ###
Your goal is to evaluate if the AI architect's JSON output (`ai_output_to_verify`) is a **reasonable and defensible conclusion** based on the `source_data` and the rules in the `<original_prompt>`.

### The "Trust but Verify" Principle for Tactic Extraction ###
Your default stance is that the junior AI is correct. Your job is to find clear evidence to the contrary.

1.  **Start with the Executor's Conclusion:** Look at the executor's `selected_tactic` and `justification` fields first.
2.  **Check for Fatal Flaws:**
    * **Relevance Contradiction:** Did the executor say `is_tactic_relevant: false` but still select a tactic? This is an **INVALID** logical error.
    * **Hallucination:** Is the `selected_tactic` a value that was not in the provided list of tactics? This is **INVALID**.
3.  **Assess the Tactic's Fit (Be Receptive):**
    * If there are no fatal flaws, read the `sentence`, the executor's `core_concept_analysis`, and the definition of the `selected_tactic`.
    * Ask the key question: **Is the selected tactic a *reasonable* fit for the concept described?** It does not need to be the single best possible answer, only a logical and defensible one. If the connection makes sense, the reasoning is **VALID**.
    * **Respect "None":** If the executor correctly determined that an architectural discussion was happening but that none of the provided tactics were a good fit, its choice of `"None"` is sophisticated and **VALID**.

### Verification & Audit Process ###

1.  **Assess Executor's Reasoning:**
    * Read the executor's entire output in `<ai_output_to_verify>`.
    * Read the `source_data` and the tactic list from the `<original_prompt>`.
    * Based on the "Trust but Verify" principle, is the executor's chain of thought and final conclusion valid and defensible?
    * `is_executor_reasoning_valid`: Set to `True` if the reasoning is plausible and follows the rules; otherwise, set to `False`.

2.  **Render Final Verdict:**
    * `evaluation`: This is a direct reflection of your audit. If `is_executor_reasoning_valid` is `True`, the `evaluation` MUST be `correct`. If it is `False`, the `evaluation` MUST be `incorrect`.
    * `reasoning`: Write a concise, one-sentence justification for your verdict. Start by stating if the executor's reasoning was valid and why. (e.g., "The executor's reasoning was valid because 'Component Replacement' is a defensible tactic for the described activity." or "The executor's reasoning was invalid due to a logical contradiction: it found the text irrelevant but still selected a tactic.")

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
    TacticExtractionVerification(hostname=LLMHost.GREEN_LAB, batch_size_override=20, keep_alive="5m", disable_cache=True).execute_verification()


if __name__ == "__main__":
    main()
