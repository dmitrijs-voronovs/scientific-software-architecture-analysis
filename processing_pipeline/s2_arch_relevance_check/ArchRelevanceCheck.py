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

Evaluate whether the content explicitly discusses or relates to software architecture concepts, principles, or concerns at a system level. Your goal is to distinguish between high-level architectural discussions and low-level implementation details.

Data:

Content: {x['sentence']}
Instructions:

1.  **Analyze the content for system-level architectural topics.** These include, but are not limited to:
    *   Architectural patterns or styles (e.g., microservices, monolith, event-driven architecture).
    *   Decisions about the overall structure of a system, its layers, and the high-level interactions between its major components.
    *   System-wide quality attributes and the trade-offs involved (e.g., choosing a technology for its scalability properties).
    *   Dependencies and constraints that impact the entire system.

2.  **Identify implementation-level topics.** These include general software development, debugging, dependency issues, library-specific configurations, or the internal logic of a single algorithm or function.

3.  **Apply the following classification rule:**
    *   If the primary focus of the content is on the **system-level** topics described in Instruction 1, mark it as `related_to_arch: true`.
    *   If the content focuses on **implementation-level** topics, mark it as `related_to_arch: false`. **This is true even if it mentions a quality attribute like performance in a narrow context.** For example, a discussion about tuning a specific algorithm's parameters for a speed-vs-accuracy trade-off is considered an implementation detail.

4.  **Provide `reasoning`** that clearly explains your classification based on the rules above.
"""

def main():
    # ArchitectureRelevanceCheckStage(hostname=LLMHost.GREEN_LAB).execute(["root-project"], reverse=True)
    ArchitectureRelevanceCheckStage(hostname=LLMHost.GREEN_LAB, disable_cache=True, batch_size_override=10).execute_single_threaded(["root-project.root.v6-32-06.code_comment.", "root-project.root.v6-32-06.docs.", "root-project.root.v6-32-06.issue_comment."], reverse=True)

if __name__ == "__main__":
    main()
