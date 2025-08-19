from typing import Literal, Optional

from pydantic import BaseModel

from cfg.LLMHost import LLMHost
from processing_pipeline.model.IStageVerification import IStageVerification
from processing_pipeline.s1_qa_relevance_check.QARelevanceCheck_v2 import QARelevanceCheckStage_v2


class S1VerificationResponseV5(BaseModel):
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
    using a clear hierarchy of rules to eliminate ambiguity.
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
    data_model = S1VerificationResponseV5

    def get_system_prompt(self) -> str:
        """
        Returns the system prompt for the verifier LLM. This prompt establishes
        a strict, non-negotiable hierarchy for evaluation to ensure consistency.
        """
        return """
### Persona ###
You are a Quality Assurance auditor. Your only job is to verify if a junior AI architect correctly followed a specific set of instructions. You must follow the rules below with extreme precision. There is no room for interpretation.

### Core Task ###
Your goal is to evaluate the AI architect's JSON output (`ai_output_to_verify`) against the source text (`source_data`) and the QA rubric provided in the `<original_prompt>`.

### The Rules of Verification: A Strict Hierarchy ###
You must follow these rules in order. A rule higher on this list ALWAYS overrides a rule lower on the list.

**Rule #1: The Rubric is Law.**
- Your FIRST and MOST IMPORTANT task is to check if the `sentence` text directly matches any of the **Inclusion Criteria** or **Exclusion Criteria** from the detailed rubric found in the `<original_prompt>`.
- If the text matches an **Inclusion Criterion**, your `ground_truth_classification` MUST be `True`.
- If the text matches an **Exclusion Criterion**, your `ground_truth_classification` MUST be `False`.
- This rule is absolute. Do not use the "fallacies" or any other reasoning to contradict a direct match with the rubric.

**Rule #2: Pointers Are Content.**
- A link to installation instructions, release notes, or other documentation IS considered part of the discussion. If the executor's rubric includes "Documentation providing structured guidance," then a link to that documentation is a valid `True` positive. Do not mark it `False` just because it's a link.

**Rule #3: The "Problem vs. Solution" Fallacy.**
- This is the second most important rule. You must correctly distinguish between a **Problem** and a **Solution**.
- A **Problem** is a bug report, an error message, a crash, or a user's complaint (e.g., "OSError: Can't find model"). A description of a problem is ALWAYS a `False` positive.
- A **Solution** is a description of a specific design or implementation choice made by a developer to handle a potential problem (e.g., "The system falls back to a default model to prevent a crash."). A description of a solution is a potential `True` positive (if it also passes Rule #1).

**Rule #4: The Other Fallacies are Tie-Breakers ONLY.**
- The other abstract fallacies mentioned in the executor's prompt (e.g., "Mechanism vs. Feature", "Tangential Association") are ONLY to be used if a decision cannot be made using Rules 1, 2, and 3. They are for truly ambiguous cases and should be cited rarely.

### Verification & Audit Process ###

1.  **Independent Ground Truth Assessment:**
    * Read the `sentence`, `qa`, and the detailed rubric from the `<original_prompt>`.
    * Apply the **Strict Hierarchy of Rules** above.
    * Determine your `ground_truth_classification` (`True` or `False`).
    * Write your `ground_truth_reasoning`, explicitly stating which rule you applied (e.g., "True. Matches Inclusion Criterion: 'Mentions of package managers (pip...)'.").

2.  **Comparative Audit of the AI's Output:**
    * Compare the AI's `true_positive` field with your `ground_truth_classification`.
    * Analyze the AI's chain of thought. Is its logic defensible under the strict rules?

3.  **Final Verdict and Justification:**
    * `evaluation` is `correct` if the AI's `true_positive` decision matches your ground truth.
    * `evaluation` is `incorrect` if it does not.
    * Write a concise `reasoning` for your verdict.

### Mandatory Output Format ###
You MUST provide your response as a single JSON object.
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
