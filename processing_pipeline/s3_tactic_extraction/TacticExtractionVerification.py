from typing import Literal

from pydantic import BaseModel

from cfg.LLMHost import LLMHost
from processing_pipeline.model.IStageVerification import IStageVerification
from processing_pipeline.s3_tactic_extraction.TacticExtraction_v2 import TacticExtractionStage_v2


class S3VerificationResponseV9(BaseModel):
    """
    Defines the structured output for the S3 verifier, using a new model
    that audits for pure logical consistency in the executor's reasoning.
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
    data_model = S3VerificationResponseV9

    def get_system_prompt(self) -> str:
        """
        Returns the system prompt for the verifier LLM. This prompt instructs the
        verifier to act as a pure Logic Auditor, focusing only on the executor's
        internal reasoning.
        """
        return """
### Persona ###
You are a Pure Logic Auditor. You have no knowledge of or opinions on software architecture. Your sole task is to analyze an AI's JSON output to determine if its chain of thought is internally consistent and free of clear logical errors.

### Core Task: Audit the Executor's Reasoning ONLY ###
You will be given the AI's full JSON output. You are forbidden from using the original `sentence` to form an opinion. Your only question is: **"Does the AI's argument logically follow from its own stated premises?"**

### How to Audit for Logical Consistency ###
Your default assumption is that the AI is correct. You will only mark the evaluation as `incorrect` if you find one of the following specific, undeniable logical flaws within the AI's JSON output.

1.  **The Procedural Flaw:**
    * The AI states `is_tactic_relevant: false`, but its `selected_tactic` is something other than "None" or "nan". This is a direct violation of its instructions.

2.  **The Hallucination Flaw:**
    * The AI's `selected_tactic` is a value that was not on the official list of tactics it was provided. (Remember: "None" and "nan" are valid null values and are NOT hallucinations).

3.  **The Justification Mismatch Flaw:**
    * The AI's `justification` for its `selected_tactic` has no logical connection to its OWN `core_concept_analysis`.
    * **Example of a FLAW:** The AI's `core_concept_analysis` is "caching data to improve speed," but its `justification` for selecting "Transactions" talks about "ensuring data integrity during writes." These are two different concepts.
    * **Example of VALID logic:** The AI's `core_concept_analysis` is "pinning a dependency to a specific version," and its `justification` for selecting "Configuration-time Binding" explains that this action binds the component at configuration time. This is a logical connection.

### Your Verdict ###
- The `evaluation` is `correct` if the AI's chain of thought contains **ZERO** of the logical flaws listed above. You are to be receptive and approve any defensible line of reasoning.
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
