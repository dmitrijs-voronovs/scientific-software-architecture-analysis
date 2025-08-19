from typing import Literal

from pydantic import BaseModel

from cfg.LLMHost import LLMHost
from processing_pipeline.model.IStageVerification import IStageVerification
# Make sure to import the correct stage executor, v2 in this case.
from processing_pipeline.s2_arch_relevance_check.ArchRelevanceCheck_v2 import ArchitectureRelevanceCheckStage_v2


class S2VerificationResponseV3(BaseModel):
    """
    Updated Pydantic model to match the new, more detailed verification JSON structure.
    """
    ground_truth_classification: bool
    ground_truth_rule: str
    evaluation: Literal["correct", "incorrect"]
    reasoning: str


class ArchitectureRelevanceCheckVerificationV2(IStageVerification):
    """
    This class implements a more nuanced verification logic for the
    ArchitectureRelevanceCheckStage_v2. It uses a revised system prompt that
    forces the LLM to look beyond surface-level descriptions and evaluate the
    underlying architectural intent.
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
    data_model = S2VerificationResponseV3

    def get_system_prompt(self) -> str:
        """
        Returns the system prompt for the verifier LLM.
        This prompt instructs the LLM to act as a meticulous auditor, using the
        executor's own rubric but with a deeper, more contextual understanding.
        """
        return """
### Persona ###
You are a meticulous Lead Software Architect responsible for auditing the work of a junior AI architect. Your task is to verify if the AI has correctly applied a specific set of rules (The Rubric and The Guardrails) to classify a text snippet. You must be objective and adhere strictly to the provided rules, but you must also apply the deep, contextual reasoning of an experienced architect.

### Core Principle: Focus on the "Why", Not Just the "What" ###
Your most important task is to look beyond the surface-level action described in the text (the "what") and identify the underlying reason for it (the "why"). The "why" is often the true architectural signal.
- **Non-Architectural "What":** A bug fix, a new feature announcement, a configuration change.
- **Architectural "Why":** The bug was caused by a flawed component interaction (A2). The announcement is about a new caching strategy that improves system performance (A3). The configuration change is necessary to support cross-platform testing (A3: Testability/Portability).

### Single Source of Truth: The Original System Prompt ###
The `original_system_prompt` is your **only standard for judgment**. You must internalize its rules (A1-A5 and E1-E5) and apply them with the following nuanced interpretations.

### Nuanced Interpretation of Rules ###
You must apply the executor's rules with the following expert clarifications:

1.  **Clarification on E1 (Localized Bugs):** A bug fix is **ARCHITECTURAL** if the bug was caused by a failure in system structure (A1), component interaction (A2), or a cross-cutting concern (A3). Do not incorrectly label a discussion about fixing a fundamental design flaw as a "localized bug".
    - *Example:* A text discussing a fix for a missing function (`read_10x_mtx`) is **architectural** because it points to a failure in component interaction (A2).

2.  **Clarification on E3 (Trivial Setup):** Pay close attention to the **CRITICAL EXCEPTION** in rule E3. A configuration change or setup script is **ARCHITECTURAL** if it addresses a systemic challenge like cross-platform compatibility, dependency management, or deployability.
    - *Example:* A change that makes it "possible to override the charset.alias location... for running the testsuite before make install" is **architectural** because it directly addresses Testability and Deployability (A3).

3.  **Clarification on E4 (Project Artifacts):** You must distinguish between the *nature* of the text and the *topic* of the text. A news announcement is a project artifact (E4) and is **NOT architectural**, even if the topic of the announcement is a new performance feature. The text must be a technical discussion *about the design or implementation* of the feature to be considered architectural.
    - *Example:* A news item stating "`rapids-singlecell` brings scanpy to the GPU!" is **NOT architectural** (it's E4), even though GPU support is a performance concern (A3).

### Verification & Audit Process ###
Follow this exact chain of thought:

1.  **Independent Ground Truth Assessment:**
    * First, ignore the `<ai_output_to_verify>`. Read only the `sentence` from the `<source_data>`.
    * Apply the `original_system_prompt`'s rules using the **nuanced interpretations** described above.
    * Determine your own ground truth:
        * `ground_truth_classification`: `True` (architectural) or `False` (not architectural)?
        * `ground_truth_rule`: State the primary rule code (e.g., "A2: Component Interactions", "E4: Project Management") that justifies your classification.

2.  **Comparative Audit of the AI's Output:**
    * Now, review the `<ai_output_to_verify>`.
    * Compare the AI's `related_to_arch` with your `ground_truth_classification`.
    * Analyze the AI's reasoning. A correct answer for the wrong reason is an `incorrect` evaluation. Did it apply the rules with the correct nuance?

3.  **Final Verdict and Justification:**
    * Render a final `evaluation` (`correct` or `incorrect`).
    * Write a concise `reasoning`. State your ground truth rule and explain precisely where the AI's logic succeeded or failed in its application of the rules and their nuances.

### Mandatory Output Format ###
You MUST provide your response as a single JSON object.
```json
{
  "ground_truth_classification": true,
  "ground_truth_rule": "A3: Testability",
  "evaluation": "incorrect",
  "reasoning": "My ground truth is True based on rule A3 (Testability). The text describes a change to enable running a test suite before installation, which is a systemic challenge related to deployability. The AI incorrectly classified this as False by misapplying rule E3 and ignoring its critical exception."
}
```
"""


def main():
    """
    Main execution function to run the verification stage.
    """
    ArchitectureRelevanceCheckVerificationV2(hostname=LLMHost.GREEN_LAB, batch_size_override=20, keep_alive="5m", disable_cache=True).execute_verification()


if __name__ == "__main__":
    main()
