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
You are a meticulous software engineering expert specializing in non-functional requirements (Quality Attributes). Your task is to act as a strict gatekeeper and determine if a sentence **from a software codebase or its technical documentation** is a direct and strong example of a specific quality attribute.

### Primary Directive
Your main goal is to avoid false positives. If the connection between the content and the quality attribute is weak, indirect, or requires making significant assumptions, you must classify it as a false positive.

### Data for Evaluation

**1. Quality Attribute:** {x['qa']}

**2. Attribute Description:** {x['qa_desc']}

**3. Scope & Distinctions (Crucial Guardrails):** {x['qa_scope_hint']}

**4. Content to Analyze:** {x['sentence']}

### Instructions for Analysis

Follow this **mandatory** reasoning process step-by-step:

**Step 1: Context Check.**
First, determine if the 'Content to Analyze' is plausibly a comment from a software codebase or technical documentation.
- If it reads like a scientific abstract, a news article, or any other non-software text, it is **out of scope**. Stop here and respond with `true_positive: false` and state that the content is not from a software context.

**Step 2: Analyze Intent.**
If the context is valid, identify the primary purpose of the statement. Is it describing:
- A high-level system strategy?
- A specific algorithm or component's function?
- An optimization for speed or resource use?
- A workaround for a bug or limitation?
- A low-level code convention?

**Step 3: Compare to Scope & Distinctions.**
Reread the 'Scope & Distinctions' section. Does the intent you identified in Step 2 fall squarely within this scope?
- Explicitly check if the content is an example of what the scope says to *distinguish it from*. For example, for 'Availability', the scope says to distinguish it from component-level 'Reliability'. If the content is about component-level reliability, it is a false positive for Availability.

**Step 4: Evaluate Directness.**
Is the content a *direct* illustration, or is it a low-level detail that only *indirectly supports* the attribute?
- **Example of Indirect:** A comment saying `// Using a hash map for fast lookups` supports Performance, but it is a very low-level detail.
- **Example of Direct:** A comment saying `// This caching layer was added to reduce API latency from 500ms to 50ms` is a strong, direct example of Performance.
Your task is to identify the **direct and strong** examples.

**Step 5: Final Decision.**
Based on the strict application of the steps above, make your final decision.

### Your Response

Provide your response with only the following two fields:
- `true_positive`: `true` or `false`.
- `reasoning`: A concise explanation for your decision based on the step-by-step analysis. If it is a false positive, you must suggest the more appropriate quality attribute (e.g., Reliability, Performance, Maintainability) or state that it's out of scope.
"""

    @classmethod
    def filter_and_transform_df_before_processing(cls, df):
        qa_details = {
            "availability": {
                "desc": "Availability refers to a system's ability to mask or repair faults such that the cumulative service outage period does not exceed a required value over a specified time interval, ensuring it is ready to carry out its task when needed.",
                "scope_hint": "Focus on system-level uptime and recovery from service-affecting failures (e.g., crash recovery, failover). Distinguish from component-level 'Reliability' (e.g., a single function handling a null pointer) or 'Fault Tolerance' (e.g., a try-catch block)."
            },
            "deployability": {
                "desc": "Deployability measures the ease and speed with which a new version of the system can be delivered to and installed by its users, including the time taken for updates.",
                "scope_hint": "Focus on infrastructure, automation, and processes related to release and installation (e.g., build scripts, continuous integration, package management). Distinguish from 'Modifiability', which is about changing the code itself."
            },
            "energy efficiency": {
                "desc": "Energy efficiency, also known as 'green computing', describes how well software minimises its consumption of computing resources, thus reducing associated costs like electricity, weight, and physical footprint.",
                "scope_hint": "Focus on reducing power consumption or computational load for environmental or hardware-constraint reasons (e.g., optimizing for battery life, reducing CPU cycles for thermal management)."
            },
            "integrability": {
                "desc": "Integrability refers to the ease with which software components or distinct systems can be combined and made to work together effectively as a coherent whole, often supported by mechanisms that reduce coupling and manage dependencies.",
                "scope_hint": "Focus on APIs, component design, and dependency management that allow different parts of a system to be combined. Distinguish from 'Interoperability', which is about external systems exchanging data."
            },
            "interoperability": {
                "desc": "Interoperability is the degree to which two or more systems can usefully exchange and correctly interpret meaningful information via their interfaces within a particular context.",
                "scope_hint": "Focus on data formats, communication protocols, and standardized interfaces for exchanging information between *separate systems* (e.g., JSON/XML APIs, network protocols)."
            },
            "modifiability": {
                "desc": "Modifiability refers to the ease with which changes, such as adding, deleting, or modifying functionality, quality attributes, capacity, or technology, can be made to a system, ideally involving the fewest distinct elements.",
                "scope_hint": "Focus on code structure that makes changes easier (e.g., modularity, low coupling, design patterns). A comment about refactoring or simplifying a complex function is a good example."
            },
            "performance": {
                "desc": "Performance is a system's ability to meet its timing requirements, encompassing its time-based response to events and its efficiency in resource usage under specified conditions.",
                "scope_hint": "Focus on speed, latency, throughput, and resource usage (e.g., algorithm optimization, caching, efficient memory access, concurrency to improve speed)."
            },
            "reliability": {
                "desc": "Reliability describes the degree to which a system, product, or component performs its specified functions under defined conditions for a given period, often closely related to the broader concept of availability.",
                "scope_hint": "Focus on correctness and preventing failures (e.g., error handling, input validation, managing race conditions, resource management to prevent leaks). This is often at a component or function level."
            },
            "safety": {
                "desc": "Safety refers to the software's ability to avoid entering hazardous states that could cause damage, injury, or loss of life, and to recover or limit harm if such states are entered.",
                "scope_hint": "Focus on preventing physical harm or catastrophic failure. This is common in embedded, automotive, or aerospace systems (e.g., system shutdown procedures, handling sensor failure)."
            },
            "security": {
                "desc": "Security is the degree to which a system protects information and data from unauthorised access or manipulation, ensuring confidentiality, integrity, and availability for legitimate users.",
                "scope_hint": "Focus on protection against malicious actors (e.g., preventing buffer overflows, input sanitization to stop injection attacks, authentication, encryption)."
            },
            "testability": {
                "desc": "Testability refers to the ease with which software can be made to quickly reveal its faults through execution-based testing, by providing controllability and observability of its state and limiting complexity.",
                "scope_hint": "Focus on code or design that simplifies testing (e.g., dependency injection, clear separation of concerns, adding logging or diagnostics specifically for testing purposes)."
            },
            "usability": {
                "desc": "Usability is concerned with how easily users can accomplish desired tasks and the kind of user support the system provides to facilitate their effectiveness, efficiency, and satisfaction.",
                "scope_hint": "Focus on the end-user experience (e.g., clear user interfaces, helpful error messages, intuitive workflow, accessibility features)."
            }
        }

        df["qa_desc"] = df["qa"].apply(lambda x: qa_details[x]["desc"])
        df["qa_scope_hint"] = df["qa"].apply(lambda x: qa_details[x]["scope_hint"])
        return df

    @classmethod
    def transform_df_before_saving(cls, df):
        return df.drop(columns=["qa_desc", "qa_scope_hint"])

def main():
    # QARelevanceCheckStageI(hostname=LLMHost.GREEN_LAB).execute(["root-project.root.v6-32-06.code_comment.","root-project.root.v6-32-06.docs.","root-project.root.v6-32-06.issue_comment."], reverse=False)
    QARelevanceCheckStage(hostname=LLMHost.GREEN_LAB, disable_cache=True, batch_size_override=10).execute_single_threaded(["root-project.root.v6-32-06.code_comment.", "root-project.root.v6-32-06.docs.", "root-project.root.v6-32-06.issue_comment."], reverse=True)


if __name__ == "__main__":
    main()
