# QARelevanceCheck.py (REVISED with System/User Split)

import pandas as pd
from pydantic import BaseModel

from cfg.LLMHost import LLMHost
from cfg.ModelName import ModelName
from constants.abs_paths import AbsDirPath
from constants.foldernames import FolderNames
from processing_pipeline.model.IBaseStage import IBaseStage
class OllamaQaRelevanceResponse(BaseModel):
    analysis_context_check: str
    analysis_intent: str
    analysis_scope_match: str
    true_positive: bool
    reasoning: str


class QARelevanceCheckStage_v2(IBaseStage):
    data_model = OllamaQaRelevanceResponse
    temperature = 0.0
    model_name = ModelName.DEEPSEEK_8B
    cache_dir = AbsDirPath.CACHE / FolderNames.QA_RELEVANCE_CHECK_DIR / "v2"
    in_dir = AbsDirPath.O_S0_NOISE_FILTERING
    out_dir = AbsDirPath.S1_QA_RELEVANCE_CHECK
    stage_name = 's1'

    @classmethod
    def get_system_prompt(cls) -> str:
        """
        Sets the expert persona, core directives, and mandatory output structure.
        This is the static, unchanging part of the prompt.
        """
        return """
You are a discerning software engineering expert. Your task is to analyze if a sentence provides evidence for a specific quality attribute.

### Core Principle
A quality attribute is realized through concrete design and implementation choices. Your primary goal is to identify text that describes **how** a system is built to achieve a certain quality. The implementation detail itself is the evidence.

### Primary Directives
1.  **Focus on Evidence, Not Keywords:** A sentence is a true positive if it describes a mechanism, design choice, or implementation detail that contributes to the quality attribute.
2.  **The 'How' is Sufficient:** You do not need the text to explicitly state the 'why' (e.g., "in order to make it faster"). A description of a performance-enhancing mechanism *is* an example of performance.

### Your Response: Mandatory Chain of Thought Analysis
You must generate a response with the following fields. Complete the analysis fields **first** before making your final decision.

1.  `analysis_context_check`: Is the 'Content to Analyze' from a software context? (One-sentence assessment).
2.  `analysis_intent`: What is the primary intent? "Describing Functionality" or "Describing Quality Attribute"? (State the intent clearly).
3.  `analysis_scope_match`: Does the described mechanism contribute to the given quality attribute? (One-sentence assessment).
4.  `true_positive`: `true` or `false`. This decision must be the logical conclusion of the preceding analysis steps.
5.  `reasoning`: A final, concise summary of your decision. If false, state why the described mechanism does not align with the quality attribute.
"""

    @classmethod
    def to_prompt(cls, x: pd.Series) -> str:
        """
        Provides the specific, dynamic data for the LLM to analyze in one shot.
        This is the user prompt.
        """
        return f"""
### Data for Evaluation
**1. Quality Attribute:** {x['qa']}
**2. Attribute Description:** {x['qa_desc']}
**3. Scope & Distinctions (Examples of Mechanisms):** {x['qa_scope_hint']}
**4. Content to Analyze:** {x['sentence']}

Now, apply the analysis steps defined in your system prompt to the data provided above.
"""

    @classmethod
    def filter_and_transform_df_before_processing(cls, df):
        qa_details = {
            "availability": {
                "desc": "Availability refers to a system's ability to mask or repair faults such that the cumulative service outage period does not exceed a required value over a specified time interval, ensuring it is ready to carry out its task when needed.",
                "scope_hint": "Look for mechanisms that ensure uptime. Examples: redundant components, failover logic, health checks, or caching strategies that serve content even if a downstream service fails."
            },
            "deployability": {
                "desc": "Deployability measures the ease and speed with which a new version of the system can be delivered to and installed by its users, including the time taken for updates.",
                "scope_hint": "Look for descriptions of the release and installation process. Examples: mentions of package managers (pip, conda), containerization (Docker), build automation (scripts, makefiles), or CI/CD pipeline configurations."
            },
            "energy efficiency": {
                "desc": "Energy efficiency, also known as 'green computing', describes how well software minimises its consumption of computing resources, thus reducing associated costs like electricity, weight, and physical footprint.",
                "scope_hint": "Look for implementations that reduce resource consumption. Examples: optimizing algorithms to lower CPU cycles, minimizing memory footprints, or implementing power-saving modes."
            },
            "integrability": {
                "desc": "Integrability refers to the ease with which software components or distinct systems can be combined and made to work together effectively as a coherent whole, often supported by mechanisms that reduce coupling and manage dependencies.",
                "scope_hint": "Look for designs that facilitate combining components. Examples: a 'pluggable' architecture, use of dependency injection frameworks, or a component that exposes a well-defined Service Provider Interface (SPI)."
            },
            "interoperability": {
                "desc": "Interoperability is the degree to which two or more systems can usefully exchange and correctly interpret meaningful information via their interfaces within a particular context.",
                "scope_hint": "Look for mechanisms for data exchange between separate systems. Examples: implementing a client for a specific API, using a standardized data format (JSON, XML) for communication, or adhering to a network protocol."
            },
            "modifiability": {
                "desc": "Modifiability refers to the ease with which changes, such as adding, deleting, or modifying functionality, quality attributes, capacity, or technology, can be made to a system, ideally involving the fewest distinct elements.",
                "scope_hint": "Look for code structures that make future changes easier. Examples: use of design patterns (like Strategy or Factory), creating a settings file to avoid hardcoded values, or decoupling components with an event bus."
            },
            "performance": {
                "desc": "Performance is a system's ability to meet its timing requirements, encompassing its time-based response to events and its efficiency in resource usage under specified conditions.",
                "scope_hint": "Look for mechanisms that improve speed or reduce resource usage. Examples: use of caching, pre-computation, indexing (e.g., nearest neighbor index), asynchronous processing, or memory management techniques."
            },
            "reliability": {
                "desc": "Reliability describes the degree to which a system, product, or component performs its specified functions under defined conditions for a given period, often closely related to the broader concept of availability.",
                "scope_hint": "Look for code that handles errors and edge cases within a component. Examples: null checks, exception handling blocks (try/except), input validation, or retrying a failed operation."
            },
            "safety": {
                "desc": "Safety refers to the software's ability to avoid entering hazardous states that could cause damage, injury, or loss of life, and to recover or limit harm if such states are entered.",
                "scope_hint": "Look for mechanisms that prevent real-world harm. Examples: sanity checks on dangerous operations, failsafes in physical system controllers, or features that prevent data corruption with irreversible consequences."
            },
            "security": {
                "desc": "Security is the degree to which a system protects information and data from unauthorised access or manipulation, ensuring confidentiality, integrity, and availability for legitimate users.",
                "scope_hint": "Look for mechanisms that defend against threats. Examples: input sanitization to prevent injection, use of encryption libraries, implementation of authentication/authorization checks, or protection against replay attacks."
            },
            "testability": {
                "desc": "Testability refers to the ease with which software can be made to quickly reveal its faults through execution-based testing, by providing controllability and observability of its state and limiting complexity.",
                "scope_hint": "Look for designs or features that simplify verification. Examples: implementing dependency injection to allow for mocking, adding extensive logging, creating internal APIs specifically for test harnesses, or separating concerns."
            },
            "usability": {
                "desc": "Usability is concerned with how easily users can accomplish desired tasks and the kind of user support the system provides to facilitate their effectiveness, efficiency, and satisfaction.",
                "scope_hint": "Look for features that directly improve the end-user experience. Examples: adding helpful error messages, providing default configurations, creating a graphical user interface (GUI), or adding shortcuts and tooltips."
            }
        }

        df["qa_desc"] = df["qa"].apply(lambda x: qa_details[x]["desc"])
        df["qa_scope_hint"] = df["qa"].apply(lambda x: qa_details[x]["scope_hint"])
        return df

    @classmethod
    def transform_df_before_saving(cls, df):
        return df.drop(columns=["qa_desc", "qa_scope_hint"])


def main():
    # QARelevanceCheckStage_v2(hostname=LLMHost.SERVER).execute(["root-project.root.v6-32-06.code_comment.","root-project.root.v6-32-06.docs.","root-project.root.v6-32-06.issue."], reverse=False)
    QARelevanceCheckStage_v2(hostname=LLMHost.SERVER, disable_cache=True, batch_size_override=20, n_threads_override=5).execute(["root-project"], reverse=True)


if __name__ == "__main__":
    main()