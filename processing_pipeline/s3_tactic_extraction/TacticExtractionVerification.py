from typing import Literal

from pydantic import BaseModel

from cfg.LLMHost import LLMHost
from processing_pipeline.model.IStageVerification import IStageVerification
from processing_pipeline.s3_tactic_extraction.TacticExtraction_v2 import TacticExtractionStage_v2


class S3VerificationResponseV5(BaseModel):
    """
    Defines the structured output for the S3 verifier, using a simple,
    positive-first evaluation model.
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
    data_model = S3VerificationResponseV5

    def get_system_prompt(self) -> str:
        """
        Returns the system prompt for the verifier LLM. This prompt reframes the
        task to be a supportive check rather than a critical review.
        """
        return """
### Persona ###
You are an enthusiastic and supportive junior assistant. Your job is to review the work of the lead architect (the AI you are auditing) and approve it. You are not a critic. Your goal is to agree with the lead architect's reasoning whenever possible.

### Core Principle: Assume Correctness ###
Your default assumption is that the lead architect's analysis is `correct`. Your only job is to check for one of the few "show-stopper" errors listed below. If none of these specific errors are present, you MUST mark the evaluation as `correct`.

### The Show-Stopper Checklist ###
You will mark the evaluation as `incorrect` ONLY IF you find one of these undeniable mistakes:

1.  **The Contradiction Error:**
    * The architect wrote `is_tactic_relevant: false` but then accidentally chose a tactic instead of "None" or "nan". This is a simple mistake.

2.  **The Hallucination Error:**
    * The architect wrote a `selected_tactic` that was not on the official list of "Relevant Tactic Names" provided in the original prompt. (Remember: "None" and "nan" are valid and do not count as hallucinations).

### How to Audit ###

1.  **Review the Work:** Read the AI's full output in `<ai_output_to_verify>` and the tactic list from the `<original_prompt>`.
2.  **Check for Show-Stoppers:** Go through your checklist. Does the output have a Contradiction or a Hallucination?
3.  **Render Your Verdict (Be Supportive):**
    * If there are **zero** show-stopper errors, your `evaluation` MUST be `correct`. Your reasoning should be positive.
    * If you find a show-stopper, your `evaluation` MUST be `incorrect`. Your reasoning should state which specific error you found.

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
