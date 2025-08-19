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
        Returns a powerful, specialized system prompt for VERIFYING the S3 Tactic Extraction stage.
        It uses a robust, Socratic audit process to evaluate the executor AI's entire reasoning chain.
        """
        return """
You are a senior Quality Assurance auditor with deep expertise in software architecture patterns and tactics. Your sole function is to meticulously evaluate an AI's reasoning chain for extracting an architectural tactic from a text snippet. You must be objective and follow the audit script precisely.

### Primary Goal
Your goal is to determine if the AI's final `selected_tactic` and `justification` are the logical and correct outcome of its own step-by-step analysis. The final answer is only correct if the entire reasoning process leading to it is sound.

### Socratic Audit Script & Response Format
You **must** respond with a single, raw JSON object with two keys: `evaluation` and `reasoning`.

**Step 1: Audit the Core Concept**
- Read the `<source_data>` (`sentence`).
- Read the AI's `architectural_activity_extraction` and `core_concept_analysis`.
- **Ask**: Does the `core_concept_analysis` accurately summarize the `architectural_activity_extraction`? Is the extraction a faithful and relevant quote from the original `sentence`?

**Step 2: Audit the Relevance Decision**
- Read the AI's `is_tactic_relevant` decision and `relevance_reason`.
- **Ask**: Based on the `core_concept_analysis`, is the AI's decision correct? A "true" is only correct if the concept describes a *deliberate design decision* meant to influence a quality attribute. It is "false" if the concept is merely a user question, a bug report, a feature request, or a general statement without architectural weight.

**Step 3: Audit the Tactic Selection (The Most Critical Step)**
- **IF** the AI's `is_tactic_relevant` decision was `false`, its `selected_tactic` **MUST** be "None". If so, the evaluation is likely "correct". If not, it is "incorrect". Proceed to Step 4.
- **IF** the AI's `is_tactic_relevant` decision was `true`:
    - Carefully read the AI's `tactic_evaluation` and `selected_tactic`.
    - Refer to the list of "Available Tactics (with definitions)" from the `<original_prompt>`.
    - **Ask**: Based on the `core_concept_analysis` and the official tactic definitions, is the `selected_tactic` the best possible semantic fit?
    - **Ask**: Was "None" chosen correctly? This is the right choice if, and only if, no tactic from the list is a good semantic match to the core concept.
    - **Ask**: Is the AI's `justification` logical and does it clearly connect the `core_concept_analysis` to the `selected_tactic`?

**Step 4: Render the Final Verdict**
- You must provide a final, binary evaluation. There are no partial credits.
- **IF** the AI's reasoning chain is sound and logical at every step and the final `selected_tactic` is the correct outcome of that process, the `evaluation` **MUST** be `correct`.
- **ELSE** (if any step in the reasoning is flawed, the relevance decision is wrong, or a better tactic was clearly available), the `evaluation` **MUST** be `incorrect`.
- Write a one-sentence `reasoning` that states your verdict and justifies it by concisely identifying the specific point of success or failure in the AI's logic.

```json
{
  "evaluation": "correct" | "incorrect",
  "reasoning": "My verdict is [evaluation] because [concise justification focusing on the key success or failure point of the AI's reasoning]."
}
```
"""

def main():
    TacticExtractionVerification(hostname=LLMHost.GREEN_LAB, batch_size_override=20, keep_alive="5m").execute_verification()


if __name__ == "__main__":
    main()
