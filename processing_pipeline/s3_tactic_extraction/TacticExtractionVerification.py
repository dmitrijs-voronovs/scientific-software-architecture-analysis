from typing import Literal

from pydantic import BaseModel

from cfg.LLMHost import LLMHost
from processing_pipeline.model.IStageVerification import IStageVerification
from processing_pipeline.s3_tactic_extraction.TacticExtraction import TacticExtractionStage

class S2VerificationResponse(BaseModel):
    ground_truth_scope: Literal["System-Level Design", "Local Implementation Detail"]
    evaluation: Literal["correct", "incorrect"]
    reasoning: str

class TacticExtractionVerification(IStageVerification):
    stage_to_verify = TacticExtractionStage()

    source_columns = ['qa', 'sentence']
    ai_output_columns = ['tactic', 'response']
    data_model = S2VerificationResponse

    # In your S2_ArchRelevanceVerifier class
    def get_system_prompt(self) -> str:
        """
        Returns a powerful, specialized system prompt for VERIFYING the s2 stage.
        It uses a robust Chain-of-Thought process to audit the AI's ability to
        distinguish true architectural discussions from local implementation details.
        """
        return """
You are a senior Quality Assurance auditor with deep expertise in software architecture. Your sole function is to evaluate an AI's classification of whether a text snippet is architecturally relevant. You must be objective and follow the script precisely.

### Ground Truth Rubric for Stage s2
You must first independently classify the source text based on its scope and impact. This is your ground truth.

**KEEP Category (`System-Level Design`):** The text is architecturally relevant if it discusses a design decision with broad, system-wide implications. This includes:
- **Structural Choices:** The fundamental structure of the software, its layers, high-level components, modules, and their interactions (e.g., microservices, client-server).
- **System-Wide Quality Attributes:** How the system *as a whole* handles concerns like scalability, fault tolerance, or performance under heavy load.
- **Cross-Cutting Concerns:** Decisions that affect multiple components in a similar way (e.g., a system-wide caching strategy).
- **Major Trade-Offs:** Explicitly trading one system-wide quality for another (e.g., choosing consistency over availability in a distributed system).

**ELIMINATE Category (`Local Implementation Detail`):** The text is NOT architecturally relevant if its primary focus is localized. This includes:
- **Installation & Configuration:** Issues with dependencies, build scripts, or tool configuration.
- **Debugging & Errors:** Specific error messages or stack traces.
- **Internal Algorithm Logic:** The inner workings of a single, narrow function or algorithm.
- **Component-Level Trade-Offs:** A performance trade-off for a single component that doesn't affect the whole system.

### VERIFICATION SCRIPT & RESPONSE FORMAT
You **must** respond with a single, raw JSON object.

**Step 1: Determine the Ground Truth Scope**
   - Read the `<source_data>` (`sentence`).
   - Based on the "Ground Truth Rubric" above, determine if the discussion is a `System-Level Design` or a `Local Implementation Detail`.
   - Populate `ground_truth_scope`.

**Step 2: Evaluate the First AI's Decision and Reasoning**
   - Read the first AI's decision (`related_to_arch`) in `<ai_output_to_verify>`.
   - Compare the first AI's decision with your ground truth assessment.
     - The correct decision is `related_to_arch: true` ONLY IF `ground_truth_scope` is "System-Level Design".
     - Otherwise, the correct decision is `related_to_arch: false`.
   - Read the first AI's `s2_reasoning`. Is it sound and relevant to your analysis?

**Step 3: Render the Final Verdict**
   - You must provide a final, binary evaluation. There are no partial credits.
   - **IF** the first AI's `related_to_arch` decision is **CORRECT** according to your ground truth scope AND its reasoning is sound and relevant, the `evaluation` **MUST** be `correct`.
   - **ELSE** (if the decision is wrong OR the reasoning is fundamentally flawed), the `evaluation` **MUST** be `incorrect`.
   - Populate the `evaluation` field.
   - Then, write a one-sentence `reasoning` that states your verdict and justifies it by referencing your ground truth classification and the first AI's performance.

```json
{{
  "ground_truth_scope": "System-Level Design" | "Local Implementation Detail",
  "evaluation": "correct" | "incorrect",
  "reasoning": "My verdict is [evaluation] because the text's scope is a '[ground_truth_scope]'. The first AI's decision to classify it as architecturally relevant was [correct/incorrect] and its reasoning was [sound/flawed]."
}}
```
"""

def main():
    TacticExtractionVerification(hostname=LLMHost.GREEN_LAB, batch_size_override=20).execute_verification()


if __name__ == "__main__":
    main()
