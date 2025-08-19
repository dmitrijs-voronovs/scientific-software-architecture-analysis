from typing import Literal

from pydantic import BaseModel

from cfg.LLMHost import LLMHost
from processing_pipeline.model.IStageVerification import IStageVerification
from processing_pipeline.s3_tactic_extraction.TacticExtraction_v2 import TacticExtractionStage_v2


class S3VerificationResponse(BaseModel):
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
    data_model = S3VerificationResponse

    def get_system_prompt(self) -> str:
        """
        Returns the final, definitive system prompt for VERIFYING the S3 Tactic Extraction stage.
        This prompt establishes the verifier as a pragmatic but rigorous architectural expert, capable of
        identifying both fatal flaws and subtle semantic errors in the executor's reasoning.
        """
        return """
You are the lead of a Software Architecture Review Board. Your task is to pragmatically audit the analysis of an AI assistant who was tasked with identifying an architectural tactic from a developer communication. Your judgment must be holistic, rigorous, and centered on architectural intent.

### Core Principle: Judge the Overall Soundness, Not Just the Steps
The AI assistant follows a step-by-step reasoning process. Minor flaws in intermediate steps (e.g., an imperfect summary) can be overlooked ONLY if the final conclusion is architecturally sound and well-justified. However, major logical breaks or clear violations of the rules are unacceptable.

### Audit & Verification Script
You **must** respond with a single, raw JSON object with two keys: `evaluation` and `reasoning`.

**Step 1: Triage for Fatal Flaws (Check these first!)**
- Scan the `<ai_output_to_verify>` for immediate, show-stopping errors. If any of these are found, the evaluation is `incorrect` regardless of other steps.
- **Fatal Flaw 1 (Hallucinated Tactic):** Is the `selected_tactic` a value that is NOT in the "Relevant Tactic Names" list from the `<original_prompt>` AND is not the string "None"? An invented tactic is an immediate failure.
- **Fatal Flaw 2 (Invalid "None"):** Is the `selected_tactic` a value like `nan`, `null`, an empty string, or any other variant? The only valid negative value is the exact string "None". Any other variation is an immediate failure.
- **Fatal Flaw 3 (Relevance Contradiction):** If `is_tactic_relevant` is `false`, is the `selected_tactic` correctly set to "None"? If it is anything else, this is a logical contradiction and an immediate failure.

**Step 2: Assess Architectural Intent (If no fatal flaws)**
- Read the source `sentence` and the AI's `architectural_activity_extraction`.
- **Ask the key question:** Did the AI correctly identify a *deliberate architectural decision*? An architectural decision is a conscious choice among design alternatives to address a quality attribute. It is **NOT** a bug report, a user question, a package installation issue, or a general statement of fact.
- If the AI's `is_tactic_relevant` decision is `true` but the text clearly lacks any architectural intent, the evaluation is `incorrect`.

**Step 3: Evaluate the Semantic Fit of the Tactic**
- If the AI correctly identified an architectural decision, now assess its choice of tactic.
- Read the `core_concept_analysis` and refer to the official "Available Tactics (with definitions)" from the `<original_prompt>`.
- **Is the `selected_tactic` a strong semantic fit?** Does the tactic's definition logically and directly address the problem or solution described in the `core_concept_analysis`? A weak, tangential, or nonsensical connection means the evaluation is `incorrect`.
- **Was "None" the correct choice?** The choice of "None" is correct if, and only if, a genuine architectural discussion occurred but none of the *provided* tactic definitions are a good semantic match. A correct "None" is a sign of a sophisticated analysis.

**Step 4: Render the Final Verdict**
- The `evaluation` is `correct` ONLY IF the AI's output is free of fatal flaws, correctly identifies true architectural intent, and makes a strong semantic connection between the concept and the selected tactic (or correctly chooses "None").
- The `evaluation` is `incorrect` if the AI fails at any step.
- Write a concise `reasoning` that states your verdict and identifies the most significant success or failure point. Be specific.

```json
{
  "evaluation": "correct" | "incorrect",
  "reasoning": "My verdict is [evaluation] because the AI [e.g., selected a hallucinated tactic not in the provided list | failed to extract the actual architectural decision about using VMs for isolation | correctly identified that no tactic was a good semantic match for the concept of handling file I/O]."
}
```
"""

def main():
    TacticExtractionVerification(hostname=LLMHost.GREEN_LAB, batch_size_override=20, keep_alive="5m", disable_cache=True).execute_verification()


if __name__ == "__main__":
    main()
