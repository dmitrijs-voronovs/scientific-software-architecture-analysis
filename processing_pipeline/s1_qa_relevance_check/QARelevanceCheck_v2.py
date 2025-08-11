import pandas as pd
from pydantic import BaseModel

from cfg.LLMHost import LLMHost
from cfg.ModelName import ModelName
from constants.abs_paths import AbsDirPath
from constants.foldernames import FolderNames
from processing_pipeline.model.IBaseStage import IBaseStage
class OllamaQaRelevanceResponse(BaseModel):
    true_positive: bool
    reasoning: str


class QARelevanceCheckStage(IBaseStage):
    data_model = OllamaQaRelevanceResponse
    temperature = 0.0
    model_name = ModelName.DEEPSEEK_8B
    cache_dir = AbsDirPath.CACHE / FolderNames.QA_RELEVANCE_CHECK_DIR
    in_dir = AbsDirPath.O_S0_NOISE_FILTERING
    out_dir = AbsDirPath.S1_QA_RELEVANCE_CHECK
    stage_name = 's1'

    @classmethod
    def to_prompt(cls, x: pd.Series) -> str:
        return f"""
You are a meticulous software engineering expert acting as a strict quality gatekeeper. Your task is to determine if a sentence from a software codebase or its technical documentation is a **direct and unambiguous** example of a specific quality attribute.

### Primary Directives
1.  **Avoid False Positives at All Costs:** If a connection is weak, indirect, or requires significant assumptions, you must classify it as a false positive.
2.  **Principle of Direct Evidence:** The content must explicitly describe the *'why'* behind a design choice as it relates to a non-functional goal. Do not infer a quality attribute from a simple description of what the code *does*.

### Data for Evaluation

**1. Quality Attribute:** {x['qa']}

**2. Attribute Description:** {x['qa_desc']}

**3. Scope & Distinctions (Crucial Guardrails):** {x['qa_scope_hint']}

**4. Content to Analyze:** {x['sentence']}

### Instructions for Analysis

Follow this **mandatory** reasoning process step-by-step:

**Step 1: Context Check.**
First, determine if the 'Content to Analyze' is plausibly a comment from a software codebase or technical documentation.
- If it reads like a scientific abstract, a news article, or any other non-software text, it is **out of scope**. Respond immediately with `true_positive: false` and reasoning that the content is not from a software context.

**Step 2: Intent vs. Quality.**
Analyze the content's primary intent. This is the most important step. You must rigorously differentiate between:
- **Describing Functionality:** Text that only explains what the code *does*. (e.g., "This function parses a file.") This is **not** a true positive.
- **Describing a Quality Attribute:** Text that explains *why* the code is designed in a certain way to achieve a non-functional goal. (e.g., "The parser uses a streaming API to handle large files without running out of memory.") This **is** a potential true positive.

**Step 3: Apply the Scope & Distinctions.**
Reread the 'Scope & Distinctions' section carefully. Does the intent you identified in Step 2 fall squarely within this scope?
- A strong example must match the positive descriptions provided in the scope hint, not the concepts it should be distinguished from.

**Step 4: Final Decision.**
Based on the strict application of the steps above, make your final decision.

### Your Response

Provide your response with only the following two fields:
- `true_positive`: `true` or `false`.
- `reasoning`: A concise explanation for your decision based on the step-by-step analysis. Your explanation must begin by explicitly stating if the primary intent is 'Describing Functionality' or 'Describing a Quality Attribute'. If false, you must state which rule or distinction it failed and suggest the more appropriate quality attribute (e.g., Reliability, Performance, Maintainability).
"""
    @classmethod
    def filter_and_transform_df_before_processing(cls, df):
        qa_details = {
            "availability": {
                "desc": "Availability refers to a system's ability to mask or repair faults such that the cumulative service outage period does not exceed a required value over a specified time interval, ensuring it is ready to carry out its task when needed.",
                "scope_hint": "Focus on system-level uptime and recovery from major failures. Strong examples will describe mechanisms for handling crashes, network outages, or service failover. Distinguish this from component-level 'Reliability', which involves a single function handling bad data or preventing a null pointer error."
            },
            "deployability": {
                "desc": "Deployability measures the ease and speed with which a new version of the system can be delivered to and installed by its users, including the time taken for updates.",
                "scope_hint": "Focus on infrastructure, automation, and processes related to release and installation. Strong examples will mention build scripts, package managers (like pip or conda), Dockerfiles, or CI/CD pipelines. Distinguish this from 'Modifiability', which is about the ease of changing the code itself."
            },
            "energy efficiency": {
                "desc": "Energy efficiency, also known as 'green computing', describes how well software minimises its consumption of computing resources, thus reducing associated costs like electricity, weight, and physical footprint.",
                "scope_hint": "Focus on minimizing the consumption of computing resources. Strong examples will explicitly mention reducing power draw, optimizing for battery life, or lowering CPU/memory usage for thermal or environmental reasons."
            },
            "integrability": {
                "desc": "Integrability refers to the ease with which software components or distinct systems can be combined and made to work together effectively as a coherent whole, often supported by mechanisms that reduce coupling and manage dependencies.",
                "scope_hint": "Focus on APIs, component design, and dependency management that allow different parts of a system to be combined. Strong examples will describe how a component is designed to be pluggable or how it uses a well-defined API to connect with another component. Distinguish from 'Interoperability', which is about exchanging data with external systems."
            },
            "interoperability": {
                "desc": "Interoperability is the degree to which two or more systems can usefully exchange and correctly interpret meaningful information via their interfaces within a particular context.",
                "scope_hint": "Focus on data formats and protocols for exchanging information between *separate systems*. Strong examples will mention a standardized data format (like JSON, XML, TSV) or a network protocol for communication with an external system."
            },
            "modifiability": {
                "desc": "Modifiability refers to the ease with which changes, such as adding, deleting, or modifying functionality, quality attributes, capacity, or technology, can be made to a system, ideally involving the fewest distinct elements.",
                "scope_hint": "Focus on code structure that makes future changes easier. Strong examples will mention refactoring, decoupling, modularity, or using a design pattern for the explicit purpose of simplifying future development."
            },
            "performance": {
                "desc": "Performance is a system's ability to meet its timing requirements, encompassing its time-based response to events and its efficiency in resource usage under specified conditions.",
                "scope_hint": "Focus on speed, latency, throughput, and resource usage. Strong examples will explicitly mention speed (e.g., 'faster'), time (e.g., 'reduces latency'), or resource usage (e.g., 'uses less memory')."
            },
            "reliability": {
                "desc": "Reliability describes the degree to which a system, product, or component performs its specified functions under defined conditions for a given period, often closely related to the broader concept of availability.",
                "scope_hint": "Focus on correctness and preventing failures at the component or function level. Strong examples will describe handling an error, an edge case, or invalid input to prevent a single component from crashing or producing incorrect output."
            },
            "safety": {
                "desc": "Safety refers to the software's ability to avoid entering hazardous states that could cause damage, injury, or loss of life, and to recover or limit harm if such states are entered.",
                "scope_hint": "Focus on preventing physical harm or catastrophic failure. Strong examples will relate to systems where a failure could cause real-world harm (e.g., medical, automotive) and describe a mechanism to prevent that specific harm."
            },
            "security": {
                "desc": "Security is the degree to which a system protects information and data from unauthorised access or manipulation, ensuring confidentiality, integrity, and availability for legitimate users.",
                "scope_hint": "Focus on protection against malicious actors. Strong examples will explicitly mention a security threat (e.g., injection attack, unauthorized access) or a security mechanism (e.g., encryption, input sanitization, authentication)."
            },
            "testability": {
                "desc": "Testability refers to the ease with which software can be made to quickly reveal its faults through execution-based testing, by providing controllability and observability of its state and limiting complexity.",
                "scope_hint": "Focus on code or design that simplifies testing. Strong examples will mention a specific testing practice (like dependency injection, mocking) or adding a feature (like logging) for the explicit purpose of making testing easier."
            },
            "usability": {
                "desc": "Usability is concerned with how easily users can accomplish desired tasks and the kind of user support the system provides to facilitate their effectiveness, efficiency, and satisfaction.",
                "scope_hint": "Focus on the end-user experience. Strong examples will describe something that makes the software easier for a human to use, such as a clearer user interface, a more helpful error message, or a more intuitive workflow."
            }
        }

        df["qa_desc"] = df["qa"].apply(lambda x: qa_details[x]["desc"])
        df["qa_scope_hint"] = df["qa"].apply(lambda x: qa_details[x]["scope_hint"])
        return df

    @classmethod
    def transform_df_before_saving(cls, df):
        return df.drop(columns=["qa_desc", "qa_scope_hint"])


def main():
    QARelevanceCheckStage(hostname=LLMHost.SERVER).execute(["root-project.root.v6-32-06.code_comment.","root-project.root.v6-32-06.docs.","root-project.root.v6-32-06.issue_comment."], reverse=False)


if __name__ == "__main__":
    main()