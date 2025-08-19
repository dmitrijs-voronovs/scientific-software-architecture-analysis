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
        This version is more lenient on minor formatting but remains strict on core architectural logic.
        """
        return """
You are a senior Software Architecture expert acting as a pragmatic peer reviewer. Your goal is to audit an AI's reasoning for extracting an architectural tactic. Be fair and focus on whether the final outcome is architecturally sound, even if the intermediate steps aren't perfect.

### Core Principle: Judge Overall Soundness, Not Minor Flaws
Your primary goal is to assess if the AI's final `selected_tactic` is a **reasonable and defensible interpretation** of the source text. Do not fail an evaluation for minor imperfections (e.g., an imperfect summary) if the final conclusion is logical.

### Audit & Verification Script
You **must** respond with a single, raw JSON object with two keys: `evaluation` and `reasoning`.

**Step 1: Triage for Fatal Flaws**
- Scan for show-stopping errors that make the output immediately `incorrect`.
- **Fatal Flaw 1 (Hallucinated Tactic):** Is the `selected_tactic` a value that is NOT in the "Relevant Tactic Names" list and is also not a null value (e.g., "None", "nan")? An invented tactic is an immediate failure.
- **Fatal Flaw 2 (Relevance Contradiction):** If `is_tactic_relevant` is `false`, is the `selected_tactic` correctly set to a null value ("None" or "nan")? If it has a tactic name, this is a logical contradiction and an immediate failure.
- *If a fatal flaw is found, report `incorrect` and state the specific violation.*

**Step 2: Assess Architectural Intent (If no fatal flaws)**
- Read the source `sentence` and the AI's `architectural_activity_extraction`.
- **Ask the key question:** Did the AI correctly identify a *deliberate architectural decision*? An architectural decision is a conscious choice to address a quality attribute. It is **NOT** a bug report, user question, installation problem, or general comment.
- If `is_tactic_relevant` is `true` but the text clearly lacks architectural intent, the evaluation is `incorrect`.

**Step 3: Evaluate the Semantic Fit of the Tactic**
- If an architectural decision was correctly identified, now assess the tactic choice.
- Read the `core_concept_analysis` and refer to the official "Available Tactics (with definitions)".
- **Is the `selected_tactic` a strong semantic fit?** Does the tactic's definition logically address the problem described? A weak or nonsensical connection is an `incorrect` evaluation.
- **Was a null value ("None" or "nan") the correct choice?** This is the correct choice if a real architectural discussion occurred, but none of the *provided* tactics are a good semantic match. A correct null choice is a sign of a sophisticated analysis.

**Step 4: Render the Final Verdict**
- The `evaluation` is `correct` ONLY IF the AI's output is free of fatal flaws, correctly identifies architectural intent, and makes a strong semantic connection between the concept and the selected tactic (or correctly chooses a null value).
- Otherwise, the `evaluation` is `incorrect`.
- Write a concise `reasoning` that identifies the most significant success or failure point.

```json
{
  "evaluation": "correct" | "incorrect",
  "reasoning": "My verdict is [evaluation] because the AI [e.g., selected a hallucinated tactic | failed to extract the actual architectural decision | correctly identified that no tactic was a good semantic fit]."
}
```
"""

def main():
    TacticExtractionVerification(hostname=LLMHost.GREEN_LAB, batch_size_override=20, keep_alive="5m", disable_cache=True).execute_verification()


if __name__ == "__main__":
    main()
