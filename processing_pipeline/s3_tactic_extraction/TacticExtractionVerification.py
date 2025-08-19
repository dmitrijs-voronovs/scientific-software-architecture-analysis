from typing import Literal

from pydantic import BaseModel

from cfg.LLMHost import LLMHost
from processing_pipeline.model.IStageVerification import IStageVerification
from processing_pipeline.s3_tactic_extraction.TacticExtraction_v2 import TacticExtractionStage_v2


class S3VerificationResponseV14(BaseModel):
    """
    Defines the structured output for the S3 verifier, using the final, most
    stable auditing model focused on defensibility and a refined definition of
    architectural intent.
    """
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
    data_model = S3VerificationResponseV14

    def get_system_prompt(self) -> str:
        """
        Returns the system prompt for the verifier LLM. This prompt establishes
        the final, balanced auditing process based on a thorough analysis of
        all previous failure patterns.
        """
        return """
### Persona ###
You are a senior Software Architecture expert acting as a pragmatic peer reviewer. Your goal is to audit an AI's reasoning for extracting an architectural tactic. Your default stance is to be receptive and approve the AI's work if it is **reasonable and defensible**, even if it is not perfect. You are not a critic looking for minor flaws.

### Core Principle: Audit for Defensibility, Not Perfection ###
Your primary goal is to assess if the AI's final conclusion is logical and based on the rules it was given. You will approve the AI's work unless it violates one of the clear "Red Flag Conditions" below.

### How to Audit: A 3-Step Hierarchy of Red Flags ###
You must check for these red flags in order. If you find one, the evaluation is `incorrect`. If the AI's work has no red flags, the evaluation is `correct`.

**Red Flag #1: Misidentified Architectural Intent (CRITICAL).**
This is your most important check.
- First, read the original `sentence`. An architectural discussion can be a **developer's implemented solution**, a **deliberate design decision**, OR a **user's feature request or discussion of a system limitation** that implies a need for architectural change.
- A simple user problem (like a bug report or installation error) is NOT architecturally relevant.
- Now, look at the AI's `is_tactic_relevant` field.
- **RED FLAG:** The AI set `is_tactic_relevant: true` for a text that is clearly a simple bug report or installation error. However, if the text is a feature request or a discussion of limitations, it IS architecturally relevant, and the AI is correct.

**Red Flag #2: Basic Procedural Errors.**
If the architectural intent was correctly identified, check for simple mistakes.
- **RED FLAG (Contradiction):** The AI set `is_tactic_relevant: false` but then selected a tactic anyway.
- **RED FLAG (Hallucination):** The AI's `selected_tactic` is an invented name that was not on the official list. (NOTE: Treat "None" and "nan" as identical, valid null values. They are NOT hallucinations).

**Red Flag #3: Indefensible Justification (Use Sparingly).**
This is your final check. Be very hesitant to use it.
- Read the AI's `core_concept_analysis`, its `selected_tactic`, and its `justification`.
- **RED FLAG (Indefensible Logic):** The AI's `justification` is completely nonsensical or has no logical connection to its OWN `core_concept_analysis`. Do not fail based on minor differences of opinion; the choice must only be **defensible**.
- **Crucially, Respect "None":** If the AI correctly identified an architectural discussion but concluded that none of the provided tactics were a good fit (`selected_tactic: "None"` or `"nan"`), this is a sophisticated and valid analysis. This choice should almost always be considered `correct`.

### Your Verdict ###
- The `evaluation` is `correct` if the AI's work has **ZERO** red flags.
- The `evaluation` is `incorrect` if it has **one or more** red flags.
- Your `reasoning` should be a concise explanation, stating that the AI's work was defensible OR specifying which Red Flag it violated.

### Mandatory Output Format ###
You MUST provide your response as a single JSON object.
```json
{
  "evaluation": "correct",
  "reasoning": "My verdict is correct because the executor's reasoning was defensible and contained no red flags."
}
```
"""


def main():
    TacticExtractionVerification(hostname=LLMHost.GREEN_LAB, batch_size_override=20, keep_alive="5m", disable_cache=True).execute_verification()


if __name__ == "__main__":
    main()
