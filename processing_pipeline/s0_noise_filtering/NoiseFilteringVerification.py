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
        Returns the FINAL, most robust SPECIALIZED system prompt for VERIFYING the s0 stage.
        This version uses a simplified, binary (correct/incorrect) evaluation and a
        more robust, category-based reasoning chain.
        """
        return """
You are a senior Quality Assurance auditor. Your sole function is to evaluate an AI's classification of a text snippet against a strict, pre-defined rubric. You must be objective, follow the script precisely, and produce a binary 'correct' or 'incorrect' verdict.

### Ground Truth: Functional Categories for Stage s0
You must first independently classify the source text into one of the following functional categories. This is your ground truth.

**KEEP Categories (Human-Authored):**
- `Interactive Communication`: A bug report, user question, developer discussion, or direct communication.
- `Instructional Guide/Tutorial`: A README, "how-to" guide, or step-by-step instructions.
- `API/Function Documentation`: A docstring or formal documentation describing a function, its parameters, or a command's help text.
- `Technical/Scientific Explanation`: A comment or text explaining the 'why' or 'how' behind a piece of code, an algorithm, or a scientific concept.

**ELIMINATE Categories (Machine-Generated or Boilerplate):**
- `Log File/Trace`: An automated status report from a program's execution, including build logs, stack traces, or compiler warnings.
- `Raw Data List`: A bare list of technical items (e.g., file paths, API names) without explanatory prose.
- `Boilerplate Notice`: A standard, non-project-specific legal text, like a copyright or license.

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
  "ground_truth_category": "API/Function Documentation",
  "evaluation": "correct" | "incorrect",
  "reasoning": "My verdict is [evaluation] because the ground truth category is [ground_truth_category]. The first AI's decision to [keep/eliminate] was [correct/incorrect] and its reasoning was [sound/flawed]."
}}
```
    """


def main():
    NoiseFilteringStageVerification(hostname=LLMHost.GREEN_LAB, batch_size_override=20).execute_verification()


if __name__ == "__main__":
    main()
