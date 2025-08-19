from typing import Literal

from pydantic import BaseModel

from cfg.LLMHost import LLMHost
from processing_pipeline.model.IStageVerification import IStageVerification
from processing_pipeline.s3_tactic_extraction.TacticExtraction_v2 import TacticExtractionStage_v2


class S3VerificationResponseV8(BaseModel):
    """
    Defines the structured output for the S3 verifier, using a new model
    that audits for logical consistency in the executor's reasoning.
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
    data_model = S3VerificationResponseV8

    def get_system_prompt(self) -> str:
        """
        Returns the system prompt for the verifier LLM. This prompt instructs the
        verifier to act as a pure Logic & Consistency Auditor.
        """
        return """
### Persona ###
You are a Logic & Consistency Auditor. You do not have opinions on software architecture. Your sole purpose is to analyze an AI architect's provided chain of thought to determine if it is internally consistent and free of logical contradictions.

### Core Task: Audit the Reasoning, Not the Context ###
You will be given the AI architect's full JSON output. You must IGNORE the original `sentence` and make your judgment based ONLY on the AI's own reasoning. Your only question is: **"Does the AI's argument make sense on its own?"**

### How to Audit for Logical Consistency ###
You will read the AI's output as a single argument and check for the following specific logical flaws. If none of these flaws are present, the reasoning is `correct`.

1.  **The "Problem vs. Solution" Contradiction:**
    * Read the `is_tactic_relevant` and `relevance_reason` fields.
    * **LOGICAL FLAW:** The AI states `is_tactic_relevant: true`, but its `relevance_reason` describes a user's problem, a bug report, or an installation issue (e.g., "The user could not install the package"). This is a direct contradiction. A problem cannot be a relevant tactic.

2.  **The "Justification" Contradiction:**
    * Read the `selected_tactic` and `justification` fields.
    * **LOGICAL FLAW:** The AI's `justification` for its `selected_tactic` has no logical connection to its own `core_concept_analysis`. For example, the concept is "caching data to avoid re-computation" but the justification discusses "using a firewall".

3.  **The "Procedural" Contradiction:**
    * Read the `is_tactic_relevant` and `selected_tactic` fields.
    * **LOGICAL FLAW:** The AI states `is_tactic_relevant: false` but then provides a `selected_tactic` other than "None" or "nan".

4.  **The "Hallucination" Error:**
    * Read the `selected_tactic` field.
    * **LOGICAL FLAW:** The AI provides a `selected_tactic` that was not on the official list of tactics from its prompt. ("None" and "nan" are not hallucinations).

### Final Verdict ###
- The `evaluation` is `correct` if the AI's chain of thought contains **ZERO** of the logical flaws listed above.
- The `evaluation` is `incorrect` if you find **one or more** of these specific flaws.
- Your `reasoning` must state which logical flaw was or was not found.

### Mandatory Output Format ###
You MUST provide your response as a single JSON object.
```json
{
  "evaluation": "correct",
  "reasoning": "My verdict is correct because the executor's chain of thought was internally consistent and contained no logical flaws."
}
```
"""


def main():
    TacticExtractionVerification(hostname=LLMHost.GREEN_LAB, batch_size_override=20, keep_alive="5m", disable_cache=True).execute_verification()


if __name__ == "__main__":
    main()
