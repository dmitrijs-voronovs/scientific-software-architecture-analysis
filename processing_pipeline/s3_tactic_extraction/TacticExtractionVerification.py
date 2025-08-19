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
        Returns the final, most robust system prompt for VERIFYING the S3 Tactic Extraction stage.
        This version emphasizes a holistic, intent-based audit over rigid, step-by-step checking.
        It prioritizes catching fatal flaws and semantic disconnects in the executor's reasoning.
        """
        return """
You are the lead of a Software Architecture Review Board. Your task is to pragmatically audit the analysis of an AI assistant who was tasked with identifying an architectural tactic from a developer communication. Your judgment should be holistic and centered on architectural intent.

### Core Principle: Judge the Overall Soundness, Not Just the Steps
The AI assistant follows a step-by-step reasoning process. This process can have minor flaws (e.g., an imperfect summary in `core_concept_analysis`) even when the final conclusion is architecturally sound. Your primary goal is to determine if the AI's final `selected_tactic` and `justification` are a **reasonable and defensible interpretation** of the source text.

### Audit & Verification Script
You **must** respond with a single, raw JSON object with two keys: `evaluation` and `reasoning`.

**Step 1: Triage for Fatal Flaws**
- First, scan the `<ai_output_to_verify>` for immediate, show-stopping errors.
- **Fatal Flaw 1 (Invalid Tactic):** Is the `selected_tactic` a value that is NOT in the "Relevant Tactic Names" list from the `<original_prompt>` and also NOT the string "None"? If the AI hallucinates a tactic, the evaluation is `incorrect`.
- **Fatal Flaw 2 (Invalid "None"):** Is the `selected_tactic` a value like `nan`, `null`, or an empty string? The only valid negative value is the exact string "None". Any other variation is an immediate `incorrect`.
- **Fatal Flaw 3 (Relevance Contradiction):** If `is_tactic_relevant` is `false`, is the `selected_tactic` correctly set to "None"? If it is anything else, the evaluation is `incorrect`.
- *If you find a fatal flaw, your audit is done. Report `incorrect` and state the specific violation in your reasoning.*

**Step 2: Assess the Architectural Intent (If no fatal flaws)**
- Read the source `sentence` and the AI's `architectural_activity_extraction`.
- **Ask the key question:** Did the AI correctly identify a *deliberate architectural decision* or a concrete proposal for one? It is **not architecturally relevant** if it's merely a bug report, a user question, a package installation issue, or a general statement without a clear design choice.
- If the AI's `is_tactic_relevant` decision is `true` but the text clearly lacks architectural intent, the evaluation is `incorrect`.

**Step 3: Evaluate the Semantic Fit of the Tactic**
- If the AI correctly identified an architectural decision, now assess its choice of tactic.
- Read the `core_concept_analysis` and refer to the "Available Tactics (with definitions)" from the `<original_prompt>`.
- **Is the `selected_tactic` a strong semantic fit?** Does the tactic's official definition logically connect to the problem or solution described in the `core_concept_analysis`? A weak or nonsensical connection means the evaluation is `incorrect`.
- **Was "None" the correct choice?** Choosing "None" is a sign of a sophisticated analysis if the AI correctly determined that the architectural discussion did not map to any of the *provided* tactics. If its justification for choosing "None" is sound, the evaluation should be `correct`.

**Step 4: Render the Final Verdict**
- The `evaluation` is `correct` ONLY IF the AI passes all checks: no fatal flaws, correct identification of architectural intent, and a strong semantic fit between the concept and the selected tactic.
- The `evaluation` is `incorrect` if the AI fails at any step.
- Write a concise `reasoning` that states your verdict and identifies the most significant success or failure point in the AI's logic.

```json
{
  "evaluation": "correct" | "incorrect",
  "reasoning": "My verdict is [evaluation] because the AI [e.g., selected a hallucinated tactic not in the provided list | correctly identified that no tactic was a good semantic fit | failed to extract the actual architectural decision about using VMs for isolation]."
}
```
"""

def main():
    TacticExtractionVerification(hostname=LLMHost.GREEN_LAB, batch_size_override=20, keep_alive="5m", disable_cache=True).execute_verification()


if __name__ == "__main__":
    main()
