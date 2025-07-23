import pandas as pd
from pydantic import BaseModel

from cfg.LLMHost import LLMHost
from cfg.ModelName import ModelName
from constants.abs_paths import AbsDirPath
from constants.foldernames import FolderNames
from processing_pipeline.model.IBaseStage import IBaseStage


class OllamaArchitectureResponse(BaseModel):
    related_to_arch: bool
    reasoning: str


class ArchitectureRelevanceCheckStage(IBaseStage):
    data_model = OllamaArchitectureResponse
    temperature = 0.0
    model_name = ModelName.DEEPSEEK_8B
    cache_dir = AbsDirPath.CACHE / FolderNames.ARCH_RELEVANCE_CHECK_DIR
    in_dir = AbsDirPath.O_S1_QA_RELEVANCE_CHECK
    out_dir = AbsDirPath.S2_ARCH_RELEVANCE_CHECK
    stage_name = 's2'

    @classmethod
    def to_prompt(cls, x: pd.Series) -> str:
        return f"""
You are an expert in software architecture and software engineering. You have the necessary expertise to evaluate whether a given piece of content is related to software architecture.

Your goal is to determine if the content is relevant to **system-level** software architecture.

Data:

Content: {x['sentence']}
Instructions:

Follow these steps to arrive at your conclusion:

**Step 1: Initial Analysis**
First, identify the core topic of the content. Is it about a high-level system design, or is it about a specific, low-level problem?

**Step 2: Check for Architectural Concepts**
Analyze the content to see if it discusses system-level architectural concepts, even if it doesn't use the exact keywords. These concepts include:
*   **Architectural patterns or styles:** (e.g., microservices, monolith, event-driven architecture, client-server).
*   **System structure:** Discussions of system layers, high-level components, modules, and their interactions.
*   **System-wide quality attributes:** Discussions about how the system as a whole handles things like scalability, security, fault tolerance, maintainability, **consistency**, or performance under **heavy workloads**.
*   **Cross-cutting concerns:** System-wide decisions that affect multiple components.

**Step 3: Apply Exclusion Criteria**
The content is **NOT** related to architecture if its primary focus is on any of the following implementation-level topics:
*   Installation issues, dependency conflicts, or version compatibility.
*   Specific error messages, stack traces, or debugging.
*   The internal logic of a single, narrow algorithm or function.
*   Configuration of a specific tool or library.
*   A performance trade-off for a *single component* (e.g., "improving recall at the expense of indexing time" for one algorithm is an implementation detail).
*   The selection of a dataset for model training.

**Step 4: Final Classification and Reasoning**
Based on the steps above, make your final decision.
*   If the content is primarily about the system-level topics from Step 2 and does not fall into the exclusion criteria from Step 3, mark it as `related_to_arch: true`.
*   Otherwise, mark it as `related_to_arch: false`.
*   Provide `reasoning` that explicitly follows your step-by-step analysis to justify your conclusion.
"""

def main():
    ArchitectureRelevanceCheckStage(hostname=LLMHost.SERVER).execute(["code_comment.", "issue."], reverse=False)

if __name__ == "__main__":
    main()
