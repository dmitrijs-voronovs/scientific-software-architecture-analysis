import pandas as pd
from pydantic import BaseModel

from cfg.LLMHost import LLMHost
from constants.abs_paths import AbsDirPath
from constants.foldernames import FolderNames
from processing_pipeline.model.BaseStage import BaseStage


class OllamaQaRelevanceResponse(BaseModel):
    true_positive: bool
    reasoning: str


class QARelevanceCheckStage(BaseStage):
    data_model = OllamaQaRelevanceResponse
    temperature = 0.0
    model_name = "deepseek-r1:8b"
    cache_dir = AbsDirPath.CACHE / FolderNames.QA_RELEVANCE_CHECK_DIR
    in_dir = AbsDirPath.S0_NOISE_FILTERING
    out_dir = AbsDirPath.S1_QA_RELEVANCE_CHECK
    stage_name = 's1'

    @classmethod
    def to_prompt(cls, x: pd.Series) -> str:
        return f"""
You are an expert in evaluating and categorizing quality attributes in software engineering. You possess the necessary skills to distinguish sentences that clearly relate to a given quality attribute from those that do not. 

Evaluate whether the content accurately aligns with its associated quality attribute, given the context provided. Your goal is to determine if the content makes sense in relation to the quality attribute description or if it is a false positive.

Data:

Quality Attribute: {x['qa']}
Attribute Description: {x['attribute_desc']}
Content: {x['sentence']}

Instructions: 
1. Analyze the content and the attribute description.
2. Determine if the content accurately reflects the intended quality attribute in this context.
3. If it does, label it as `true_positive: true`; if not, mark it as a `true_positive: false`.
4. If the content consists only of logs with no other text, mark it as a `true_positive: false`. 
4. If the content partially consists of logs, focus on analyzing remaining text. 
5. Add `reasoning` why the content is a true or false positive.
"""

    @classmethod
    def filter_and_transform_df_before_processing(cls, df):
        qa_descriptions = {
            'Availability': "The system's readiness to perform its function when required, focusing on reliability and recovery. It involves fault masking or repair to prevent failures, ensuring minimal cumulative downtime.",
            'Deployability': 'The capability of software to be deployed into an operational environment with predictable time and effort, including options for rollback if needed. Key aspects include automation, deployment speed, and deployment granularity.',
            'Energy Efficiency': 'The system’s ability to optimize resource use and minimize energy consumption while achieving required performance. This involves monitoring, allocation, and adaptation of resources.',
            'Integrability': 'The ease of combining the system with other systems or components, measured by integration cost and technical risks. Integrability considers the complexity and compatibility of interfaces, including syntactic, semantic, behavioral, and temporal alignment.',
            'Modifiability': 'The ease with which the system can be adapted by adding, removing, or modifying features, or adjusting to new environments. This attribute involves assessing the time, cost, and impact of changes, considering factors like coupling, cohesion, and the scope of modifications.',
            'Performance': 'The system’s capacity to meet its timing requirements, managing event handling and response times effectively. Performance focuses on reducing blocked time from resource contention and optimizing resource utilization under varying load conditions.',
            'Safety': 'The system’s ability to avoid states that could lead to harm or damage. Safety encompasses detection and handling of errors (e.g., omissions, timing, incorrect values) to prevent hazardous outcomes or mitigate potential damage.',
            'Security': 'The system’s ability to safeguard information against unauthorized access, while permitting authorized access. Security emphasizes confidentiality, integrity, and availability, using tactics to detect, prevent, and respond to attacks.',
            'Testability': 'The ease of validating software functionality through testing, enabling fault detection. This includes controlling and observing the system’s state, reducing complexity, and facilitating the creation of test cases and oracles.',
            'Usability': 'The degree to which users can effectively and efficiently accomplish tasks, including support for error recovery and user satisfaction. Usability covers ease of learning, efficient usage, and adaptability to user needs.'
        }

        df["attribute_desc"] = df["qa"].apply(lambda x: qa_descriptions[x])

        # filter out noise
        df = df[~df.s0_to_eliminate]
        return df

    @classmethod
    def transform_df_before_saving(cls, df):
        return df.drop(columns=["attribute_desc"])


def main():
    QARelevanceCheckStage(hostname=LLMHost.GREEN_LAB).execute(["root-project"], reverse=True)


if __name__ == "__main__":
    main()
