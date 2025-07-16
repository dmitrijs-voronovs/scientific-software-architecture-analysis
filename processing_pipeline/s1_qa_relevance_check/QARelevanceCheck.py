import pandas as pd
from pydantic import BaseModel

from cfg.LLMHost import LLMHost
from cfg.ModelName import ModelName
from constants.abs_paths import AbsDirPath
from constants.foldernames import FolderNames
from processing_pipeline.model.BaseStage import BaseStage


class OllamaQaRelevanceResponse(BaseModel):
    true_positive: bool
    reasoning: str


class QARelevanceCheckStage(BaseStage):
    data_model = OllamaQaRelevanceResponse
    temperature = 0.0
    model_name = ModelName.DEEPSEEK_8B
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
Attribute Description: {x['qa_desc']}
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
            "availability": "Availability refers to a system's ability to mask or repair faults such that the cumulative service outage period does not exceed a required value over a specified time interval, ensuring it is ready to carry out its task when needed.",
            "deployability": "Deployability measures the ease and speed with which a new version of the system can be delivered to and installed by its users, including the time taken for updates.",
            "energy efficiency": "Energy efficiency, also known as 'green computing', describes how well software minimises its consumption of computing resources, thus reducing associated costs like electricity, weight, and physical footprint.",
            "integrability": "Integrability refers to the ease with which software components or distinct systems can be combined and made to work together effectively as a coherent whole, often supported by mechanisms that reduce coupling and manage dependencies.",
            "interoperability": "Interoperability is the degree to which two or more systems can usefully exchange and correctly interpret meaningful information via their interfaces within a particular context.",
            "modifiability": "Modifiability refers to the ease with which changes, such as adding, deleting, or modifying functionality, quality attributes, capacity, or technology, can be made to a system, ideally involving the fewest distinct elements.",
            "performance": "Performance is a system's ability to meet its timing requirements, encompassing its time-based response to events and its efficiency in resource usage under specified conditions.",
            "reliability": "Reliability describes the degree to which a system, product, or component performs its specified functions under defined conditions for a given period, often closely related to the broader concept of availability.",
            "safety": "Safety refers to the software's ability to avoid entering hazardous states that could cause damage, injury, or loss of life, and to recover or limit harm if such states are entered.",
            "security": "Security is the degree to which a system protects information and data from unauthorised access or manipulation, ensuring confidentiality, integrity, and availability for legitimate users.",
            "testability": "Testability refers to the ease with which software can be made to quickly reveal its faults through execution-based testing, by providing controllability and observability of its state and limiting complexity.",
            "usability": "Usability is concerned with how easily users can accomplish desired tasks and the kind of user support the system provides to facilitate their effectiveness, efficiency, and satisfaction."
        }

        df["qa_desc"] = df["qa"].apply(lambda x: qa_descriptions[x])
        return df

    @classmethod
    def transform_df_before_saving(cls, df):
        return df.drop(columns=["qa_desc"])


def main():
    QARelevanceCheckStage(hostname=LLMHost.GREEN_LAB).execute(["root-project"], reverse=True)


if __name__ == "__main__":
    main()
