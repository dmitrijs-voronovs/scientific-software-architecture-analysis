from typing import Literal

from pydantic import BaseModel

from cfg.LLMHost import LLMHost
from processing_pipeline.model.IStageVerification import IStageVerification
from processing_pipeline.s3_tactic_extraction.TacticExtraction_v2 import TacticExtractionStage_v2


class S3VerificationResponseV4(BaseModel):
    """
    Defines the structured output for the S3 verifier, focusing on a pragmatic
    audit of the executor's reasoning.
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
    data_model = S3VerificationResponseV4

    def get_system_prompt(self) -> str:
        """
        Returns the system prompt for the verifier LLM. This prompt reframes the
        task to be a pragmatic audit, removing rigid, error-prone checklists.
        """
        return """
### Persona ###
You are a senior Software Architecture expert acting as a pragmatic peer reviewer. Your goal is to audit an AI's reasoning for extracting an architectural tactic. Your default stance is to be receptive to the AI's conclusion unless it has made a clear, undeniable error.

### Core Principle: Audit for Reasonableness, Not Perfection ###
Your primary goal is to assess if the AI's final `selected_tactic` and `justification` are a **reasonable and defensible interpretation** of the source text. Do not fail an evaluation for minor differences in opinion.

### How to Audit ###

1.  **Check for Obvious Errors First:**
    * **Contradiction:** Did the AI say `is_tactic_relevant: false` but then select a tactic anyway? This is a clear error.
    * **Hallucination:** Is the `selected_tactic` an invented name that was not on the official list? (Note: "None" and "nan" are valid null values, not hallucinations). This is a clear error.

2.  **Evaluate the Executor's Logic (Be Receptive):**
    * If there are no obvious errors, read the executor's entire chain of thought, from `architectural_activity_extraction` to `justification`.
    * **The Key Question:** Does the executor's final conclusion logically follow from its own analysis?
    * **Crucially, Respect "None":** The executor's most sophisticated move is to correctly identify an architectural discussion (`is_tactic_relevant: true`) but then conclude that none of the *provided tactics* are a good semantic fit, resulting in `selected_tactic: "None"`. This is almost always a **correct** and intelligent analysis. You should approve it unless the fit for a listed tactic is absolutely perfect and obvious.

3.  **Render Your Verdict:**
    * The `evaluation` is `correct` if the executor avoided obvious errors and its final conclusion is defensible and logical, based on its own analysis.
    * The `evaluation` is `incorrect` only if there is a clear, undeniable flaw (like a contradiction, hallucination, or a completely nonsensical tactic choice).

### Mandatory Output Format ###
You MUST provide your response as a single JSON object.
```json
{
  "evaluation": "correct | incorrect",
  "reasoning": "string"
}
```
"""


def main():
    TacticExtractionVerification(hostname=LLMHost.GREEN_LAB, batch_size_override=20, keep_alive="5m", disable_cache=True).execute_verification()


if __name__ == "__main__":
    main()
