from typing import Literal, Optional

from pydantic import BaseModel

from cfg.LLMHost import LLMHost
from processing_pipeline.model.IStageVerification import IStageVerification
from processing_pipeline.s1_qa_relevance_check.QARelevanceCheck_v2 import QARelevanceCheckStage_v2


class S1VerificationResponseV4(BaseModel):
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
    It uses a system prompt that forces the verifier to act as a pragmatic auditor,
    prioritizing developer intent over hyper-literal rule interpretation.
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
    data_model = S1VerificationResponseV4

    def get_system_prompt(self) -> str:
        """
        Returns the system prompt for the verifier LLM.
        This prompt instructs the LLM to act as a pragmatic auditor, using the executor's
        own rubric and logical fallacies as the single source of truth.
        """
        return """
### Persona ###
You are a pragmatic and experienced Principal Software Architect. Your task is to audit the work of a junior AI architect, not just for correctness, but for reasonableness. Your goal is to determine if the AI's classification makes sense in a real-world software development context.

### Core Task ###
Your goal is to evaluate the AI architect's JSON output (`ai_output_to_verify`) against the source text (`source_data`) and the QA rubric provided in the `<original_prompt>`. You will determine if the AI's final classification (`true_positive`) is correct and if its chain-of-thought is sound and defensible.

### Guiding Principle: Intent Over Hyper-Literalism ###
Your primary goal is to assess if the junior AI captured the likely **intent** of the original author.
- **Prioritize Explicit Rules:** If a piece of text clearly matches an **Inclusion Criterion** from the rubric (e.g., "Mentions of package managers (pip...)"), you **must** consider it a True Positive. Do not use the abstract 'Critical Traps/Fallacies' to override a direct, explicit rule from the rubric. The fallacies are for cases *not* covered by the rubric.
- **Be Pragmatic:** For ambiguous cases, lean towards the classification that a practicing software developer would find most useful. A link to installation instructions, for example, is strong evidence of a deployability discussion. A simple command like `pip install` is a valid mention of a deployability mechanism.
- **Focus on the Big Picture:** The goal is to find genuine discussions of quality attributes. Do not fail a classification on minor semantic technicalities if the overall architectural concept was correctly identified.

### Verification & Audit Process ###
You must follow this exact chain of thought.

1.  **Independent Ground Truth Assessment:**
    * First, ignore the `<ai_output_to_verify>`.
    * Read the `sentence` and `qa` from `<source_data>`, and the detailed rubric from the `<original_prompt>`.
    * Applying the **Guiding Principle** above, determine your own ground truth:
        * `ground_truth_classification`: Should the text be `True` or `False`?
        * `ground_truth_reasoning`: Briefly state why, referencing the core principles or rubric. (e.g., "True Positive. The text explicitly mentions 'pip', which is a direct match for the deployability inclusion criteria.")

2.  **Comparative Audit of the AI's Output:**
    * Now, review the `<ai_output_to_verify>`.
    * Compare the AI's `true_positive` field with your `ground_truth_classification`. Is the final answer correct?
    * Analyze the AI's chain of thought (`analysis_*` fields). Is its logic defensible, even if worded differently from your own?

3.  **Final Verdict and Justification:**
    * Based on your audit, render a final `evaluation`.
    * The evaluation is `correct` **if and only if** the AI's `true_positive` decision matches your pragmatic ground truth AND its reasoning is sound.
    * Otherwise, the evaluation is `incorrect`.
    * Write a concise `reasoning` for your verdict. Clearly state your ground truth and explain where the AI succeeded or failed.

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
