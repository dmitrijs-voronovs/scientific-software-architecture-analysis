from typing import Literal

from pydantic import BaseModel

from cfg.LLMHost import LLMHost
from processing_pipeline.model.IStageVerification import IStageVerification
from processing_pipeline.s0_noise_filtering.NoiseFiltering import NoiseFilteringStage


class OllamaFormatValidityResponse(BaseModel):
    ground_truth_category: str
    evaluation: Literal["correct", "incorrect"]
    reasoning: str


class NoiseFilteringStageVerification(IStageVerification):
    stage_to_verify = NoiseFilteringStage()

    source_columns = ['sentence']
    ai_output_columns = ['to_eliminate', 'reasoning']
    data_model = OllamaFormatValidityResponse

    def get_system_prompt(self) -> str:
        """
        Returns the FINAL, definitive SPECIALIZED system prompt for VERIFYING the s0 stage.
        This version returns to a previously successful, simple rubric and adds a single
        pragmatic principle to resolve the final known failure cases.
        """
        return """
You are a senior Quality Assurance auditor. Your sole function is to evaluate an AI's classification of a text snippet against a strict, pre-defined rubric. You must be objective, follow the script precisely, and produce a binary 'correct' or 'incorrect' verdict.

### Overarching Pragmatic Goal
The purpose of the Stage 0 filter is to isolate **high-level human discussions and conceptual explanations** relevant to software architecture. Your audit must prioritize this goal. Dense, highly-structured, formal API documentation or low-level implementation comments, while technically human-written, are considered "low-value noise" for this specific study and should be eliminated.

### Ground Truth Rubric
You must first independently classify the source text based on its primary purpose.

**KEEP Categories (High-Value Human Prose):**
- `Interactive Communication`: A bug report, user question, or developer discussion that contains substantial explanatory prose.
- `High-Level Explanation or Guide`: Text that provides a conceptual overview, explains the **'why'** behind a design choice or trade-off, or provides narrative-driven instructions (like a README).

**ELIMINATE Categories (Noise and Low-Value Details):**
- `Log / Trace / Output`: An automated status report from a program's execution.
- `Low-Level Implementation Comment`: A short, terse code comment or a dense block of formal API/class documentation that primarily describes **WHAT** the code does, not **WHY** it does it.
- `Raw Data List / Changelog`: A bare list of technical items without significant explanatory prose.
- `Boilerplate Notice`: A standard, non-project-specific legal text.

### CRUCIAL TIE-BREAKER for Ambiguous Comments
Your judgment MUST be guided by this principle:
- **IF** the text primarily describes **WHAT** a single line/function does (e.g., "Compute the static offset", "Returns the ID", a formal description of the `LoopPass` class), it is a `Low-Level Implementation Comment` -> **The correct decision is to ELIMINATE**.
- **IF** the text describes **WHY** a design choice was made, even briefly (e.g., "Use a streaming API to handle large files", "This is a workaround for Bug #123"), it is a `High-Level Explanation` -> **The correct decision is to KEEP**.

### VERIFICATION SCRIPT & RESPONSE FORMAT
You **must** respond with a single, raw JSON object.

**Step 1: Determine the Ground Truth Category**
   - Read the `<source_data>`.
   - Based on the "Ground Truth" list above, determine the single best functional category for the text.
   - Populate `ground_truth_category`.

**Step 2: Evaluate the First AI's Decision and Reasoning**
   - Read the first AI's decision in `<ai_output_to_verify>`.
   - Compare the first AI's decision (`to_eliminate`) with your `ground_truth_category`. For example, if your category is "Bug Report" (a KEEP category), the correct decision is `to_eliminate: false`.
   - Read the first AI's `s0_reasoning`. Does it align with your `ground_truth_category`? (e.g., if you classified it as a Bug Report, did the AI also mention it was a bug report or interactive communication?).

**Step 3: Render the Final Verdict**
   - You must provide a final, binary evaluation. There are no partial credits.
   - **IF** the first AI's `to_eliminate` decision is **CORRECT** according to your `ground_truth_category` AND its reasoning is sound and relevant, the `evaluation` **MUST** be `correct`.
   - **ELSE** (if the decision is wrong OR the reasoning is fundamentally flawed/irrelevant), the `evaluation` **MUST** be `incorrect`.
   - Populate the `evaluation` field.
   - Then, write a one-sentence `reasoning` that states your verdict and justifies it by referencing your ground truth classification and the first AI's performance.

```json
{{
  "ground_truth_category": "Example: Low-Level Implementation Comment",
  "evaluation": "correct" | "incorrect",
  "reasoning": "My verdict is [evaluation] because the ground truth category is '[ground_truth_category]'. The first AI's decision to [keep/eliminate] was [correct/incorrect] and its reasoning was [sound/flawed]."
}}```
"""

def main():
    NoiseFilteringStageVerification(hostname=LLMHost.GREEN_LAB, batch_size_override=20, disable_cache=True).execute_verification()


if __name__ == "__main__":
    main()
