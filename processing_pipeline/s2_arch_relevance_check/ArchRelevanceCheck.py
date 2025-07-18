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


class ArchitectureRelevanceCheckStageI(IBaseStage):
    data_model = OllamaArchitectureResponse
    temperature = 0.0
    model_name = ModelName.DEEPSEEK_8B
    cache_dir = AbsDirPath.CACHE / FolderNames.ARCH_RELEVANCE_CHECK_DIR
    in_dir = AbsDirPath.S1_QA_RELEVANCE_CHECK
    out_dir = AbsDirPath.S2_ARCH_RELEVANCE_CHECK
    stage_name = 's2'

    @classmethod
    def to_prompt(cls, x: pd.Series) -> str:
        return f"""
You are an expert in software architecture and software engineering. You have the necessary expertise to evaluate whether a given piece of content is related to software architecture.

Evaluate whether the content explicitly discusses or relates to software architecture concepts, principles, or concerns. Your goal is to determine if the content is relevant to software architecture.

Data:

Content: {x['sentence']}
Instructions:

1. Analyze the content and determine whether it is discussing software architecture, including but not limited to:
    * Architectural patterns or styles (e.g., microservices, monolith, event-driven architecture).
    * Architectural decisions, trade-offs, or quality attributes (e.g., scalability, maintainability, performance).
    * High-level system structure, interactions, dependencies, or constraints.
2. If the content clearly pertains to software architecture, mark it as `related_to_arch: true`.
3. If the content is general software development, code-level details, logs, or unrelated to architecture, mark it as `related_to_arch: false`.
4. If the content includes partial architectural relevance but is mostly about implementation details, analyze whether the relevant part is strong enough to classify it as `related_to_arch: true`.
5. Provide `reasoning` explaining why the content is classified as related on unrelated.
"""


def main():
    ArchitectureRelevanceCheckStageI(hostname=LLMHost.GREEN_LAB).execute(["root-project"], reverse=True)


if __name__ == "__main__":
    main()
