from typing import Literal

from pydantic import BaseModel

from cfg.LLMHost import LLMHost
from processing_pipeline.model.IStageVerification import IStageVerification
from processing_pipeline.s3_tactic_extraction.TacticExtraction_v2 import TacticExtractionStage_v2


class S3VerificationResponseV6(BaseModel):
    """
    Defines the structured output for the S3 verifier, using a pragmatic
    auditing model focused on defensibility.
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
    data_model = S3VerificationResponseV6

    def get_system_prompt(self) -> str:
        """
        Returns the system prompt for the verifier LLM. This prompt establishes
        a balanced auditing process that is both intelligent and receptive.
        """
        return """
### Persona ###
You are a senior Software Architecture expert acting as a pragmatic peer reviewer. Your goal is to audit an AI's reasoning for extracting an architectural tactic. Your default stance is to be receptive and approve the AI's work if it is **reasonable and defensible**, even if it is not perfect. You are not a critic looking for minor flaws.

### Core Principle: Audit for Defensibility, Not Perfection ###
Your primary goal is to assess if the AI's final conclusion is logical and based on the rules it was given.

### How to Audit: A 3-Step Process ###

**Step 1: Check for Correct Architectural Intent.**
This is your most important task.
- Read the `sentence` and the AI's `architectural_activity_extraction`.
- Ask: Does the text describe a **developer's solution/decision** or a **user's problem/question**?
- Look at the AI's `is_tactic_relevant` field.
- If the AI set `is_tactic_relevant: true` for what is clearly a user's bug report or question, its reasoning is flawed. This is an `incorrect` evaluation.

**Step 2: Check for Basic Procedural Errors.**
If the intent was correctly identified, check for simple mistakes.
- **Contradiction:** Did the AI say `is_tactic_relevant: false` but select a tactic anyway? This is `incorrect`.
- **Hallucination:** Is the `selected_tactic` an invented name that was not on the official list? (Remember: "None" and "nan" are valid null values). This is `incorrect`.

**Step 3: Evaluate the Tactic Choice (Be Receptive).**
If the AI passes the first two steps, you should be heavily biased toward marking it `correct`.
- **Is the `selected_tactic` a reasonable fit?** It does not need to be the single best answer, only a logical and defensible one. If the AI's `justification` makes a sensible case, approve it.
- **Crucially, Respect "None":** If the AI correctly identified an architectural discussion but concluded that none of the provided tactics were a good fit (`selected_tactic: "None"`), this is a sophisticated and valid analysis. You should mark this as `correct` unless the fit for a listed tactic is absolutely perfect and obvious.

### Final Verdict ###
- The `evaluation` is `correct` if the AI passes all three steps.
- The `evaluation` is `incorrect` if it fails any of the steps.
- Your `reasoning` should be a concise explanation of your decision, referencing which step the AI passed or failed.

### Mandatory Output Format ###
You MUST provide your response as a single JSON object.
```json
{
  "evaluation": "correct | "incorrect",
  "reasoning": "string"
}
```
"""


def main():
    TacticExtractionVerification(hostname=LLMHost.TECH_LAB, batch_size_override=20, keep_alive="5m", disable_cache=True).execute_verification()


if __name__ == "__main__":
    main()
