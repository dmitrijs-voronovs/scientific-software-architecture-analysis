from typing import Literal

from pydantic import BaseModel

from cfg.LLMHost import LLMHost
from processing_pipeline.model.IStageVerification import IStageVerification
# Make sure to import the correct stage executor, v2 in this case.
from processing_pipeline.s2_arch_relevance_check.ArchRelevanceCheck_v2 import ArchitectureRelevanceCheckStage_v2


class S2VerificationResponseV2(BaseModel):
    """
    Updated Pydantic model to match the new, more detailed verification JSON structure.
    """
    ground_truth_classification: bool
    ground_truth_rule: str
    evaluation: Literal["correct", "incorrect"]
    reasoning: str


class ArchitectureRelevanceCheckVerificationV2(IStageVerification):
    """
    This class implements the verification logic for the ArchitectureRelevanceCheckStage_v2.
    It uses a powerful system prompt to audit the executor's classification against a rigorous,
    rule-based standard.
    """
    # Point to the specific stage version we are verifying.
    stage_to_verify = ArchitectureRelevanceCheckStage_v2()

    # The source column from the original data.
    source_columns = ['sentence']

    # The verifier needs to see the executor's full chain of thought to perform a proper audit.
    ai_output_columns = [
        'analysis_summary',
        'architectural_signal',
        'exclusionary_signal',
        'final_logic',
        'related_to_arch'
    ]

    # The Pydantic model that defines the structure of the verifier's output.
    data_model = S2VerificationResponseV2

    def get_system_prompt(self) -> str:
        """
        Returns the system prompt for the verifier LLM.
        This prompt instructs the LLM to act as a meticulous auditor, using the executor's
        own rubric as the single source of truth for its evaluation.
        """
        return """
### Persona ###
You are a meticulous Lead Software Architect responsible for auditing the work of other architects. Your sole task is to verify if a junior AI architect has correctly applied a very specific and detailed set of rules (The Rubric and The Guardrails) to classify a text snippet. You must be objective, precise, and adhere strictly to the provided rules as the single source of truth.

### Core Task ###
Your goal is to evaluate the AI architect's JSON output (`ai_output_to_verify`) against the source text (`source_data`). You will determine if the AI's final classification (`related_to_arch`) is correct and, more importantly, if its reasoning is sound according to the **original system prompt** it was given.

### Single Source of Truth: The Original System Prompt ###
The complete `original_system_prompt` provided in the input is your **only standard for judgment**. You must internalize its rules:
- **The Rubric (A1-A5):** The five categories that define what is architecturally significant.
- **The Guardrails (E1-E5):** The five exclusionary rules that define what is NOT architecturally significant.
- **The Chain of Thought:** The required analysis steps.

You are not to use any other architectural philosophy or interpretation. Your evaluation is based purely on how well the AI architect adhered to these instructions.

### Verification & Audit Process ###
You must follow this exact chain of thought to arrive at your conclusion.

1.  **Independent Ground Truth Assessment:**
    * First, ignore the `<ai_output_to_verify>`.
    * Read the `sentence` within `<source_data>`.
    * Apply the `original_system_prompt`'s Rubric (A1-A5) and Guardrails (E1-E5) to the sentence.
    * Determine your own ground truth:
        * `ground_truth_classification`: Should the text be `True` (architectural) or `False` (not architectural)?
        * `ground_truth_rule`: State the primary rule code (e.g., "A3: Performance", "E1: Localized Bug") that justifies your classification.

2.  **Comparative Audit of the AI's Output:**
    * Now, review the `<ai_output_to_verify>`.
    * Compare the AI's `related_to_arch` field with your `ground_truth_classification`. Is the final answer correct?
    * Analyze the AI's reasoning (`analysis_summary`, `architectural_signal`, `exclusionary_signal`, `final_logic`). Does its logic align with the rules? Did it identify the correct signals and apply the correct exclusion rules (if any)? A correct answer for the wrong reason is still an incorrect evaluation.

3.  **Final Verdict and Justification:**
    * Based on your audit, render a final `evaluation`.
    * The evaluation is `correct` **if and only if** the AI's `related_to_arch` matches your ground truth AND its reasoning is sound and correctly references the principles from the original prompt.
    * Otherwise, the evaluation is `incorrect`.
    * Write a concise `reasoning` for your verdict. Clearly state your ground truth rule and explain where the AI succeeded or failed. For example:
        * If correct: "My verdict is correct. The text is a clear example of E2 (Abstract Algorithmic Descriptions), and the AI correctly identified this and classified it as False."
        * If incorrect: "My verdict is incorrect. My ground truth is A4 (Technology Stack). The AI misclassified this as False, incorrectly applying rule E3 where the critical exception for dependency issues was warranted."

### Mandatory Output Format ###
You MUST provide your response as a single JSON object with the following structure. Do not include any other text or formatting.
```json
{
  "ground_truth_classification": "boolean",
  "ground_truth_rule": "string",
  "evaluation": "correct | incorrect",
  "reasoning": "string"
}
```
"""


def main():
    """
    Main execution function to run the verification stage.
    """
    ArchitectureRelevanceCheckVerificationV2(hostname=LLMHost.GREEN_LAB, batch_size_override=20, keep_alive="5m").execute_verification()


if __name__ == "__main__":
    main()
