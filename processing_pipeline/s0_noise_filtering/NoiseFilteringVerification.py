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
        This version uses a highly refined, pragmatic Ground Truth rubric to ensure it
        judges the s0 model's output based on the true goals of the filtering task.
        """
        return """
You are a senior Quality Assurance auditor. Your sole function is to evaluate an AI's classification of a text snippet against a strict, pre-defined rubric. You must be objective, follow the script precisely, and produce a binary 'correct' or 'incorrect' verdict.

### Ground Truth: Functional Categories for Stage s0
You must first independently classify the source text into one of the following functional categories. This is your ground truth. The goal is to isolate high-level human discussions and explanations.

**KEEP Categories (High-Value Human Prose):**
- `Interactive Communication`: A bug report, user question, developer discussion, or direct communication that contains substantial explanatory prose. (e.g., "Thanks, it worked with `pip install...` but now I have this versioning issue...").
- `High-Level Technical Explanation`: A comment or text that explains the **'why' or 'how'** behind a design choice, an algorithm, or a trade-off. (e.g., "We use a streaming API here to handle large files without running out of memory.").
- `High-Level Instructional Guide`: A README or tutorial section that provides a conceptual overview or step-by-step instructions in natural language.

**ELIMINATE Categories (Noise and Low-Level Details):**
- `Log File / Trace / Terminal Output`: An automated status report from a program's execution, including build logs, stack traces, compiler warnings, or the output of shell commands.
- `Formal API/Class Documentation`: Dense, structured documentation that describes classes, functions, or parameters without a high-level narrative. (e.g., "The `LoopPass` class executes on each loop...").
- `Low-Level Implementation Note`: A short, terse code comment describing a single line or variable's function. (e.g., "Compute the static offset...", "note : start can be NULL if malloc fails !").
- `Raw Data List`: A bare list of technical items (e.g., file paths, API names) without explanatory prose.
- `Boilerplate Notice`: A standard, non-project-specific legal text, like a copyright or license.

### VERIFICATION SCRIPT & RESPONSE FORMAT
(The rest of the prompt remains identical to your current successful version: Step 1 to determine category, Step 2 to evaluate, Step 3 for the final verdict, and the same JSON output format.)

```json
{{
  "ground_truth_category": "Example: Low-Level Implementation Note",
  "evaluation": "correct" | "incorrect",
  "reasoning": "My verdict is [evaluation] because the ground truth category is '[ground_truth_category]'. The first AI's decision to [keep/eliminate] was [correct/incorrect] and its reasoning was [sound/flawed]."
}}
```
"""

def main():
    NoiseFilteringStageVerification(hostname=LLMHost.GREEN_LAB, batch_size_override=20, disable_cache=True).execute_verification()


if __name__ == "__main__":
    main()
