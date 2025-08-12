# QARelevanceCheck.py (REVISED to v3 based on expert analysis)

import pandas as pd
from pydantic import BaseModel

from cfg.LLMHost import LLMHost
from cfg.ModelName import ModelName
from constants.abs_paths import AbsDirPath
from constants.foldernames import FolderNames
from processing_pipeline.model.IBaseStage import IBaseStage


class OllamaQaRelevanceResponse(BaseModel):
    """
    Defines the structured output required from the LLM, with a more rigorous
    chain-of-thought analysis based on the findings of the error report.
    """
    analysis_problem_vs_solution: str
    analysis_mechanism_vs_feature: str
    analysis_causal_link: str
    analysis_rubric_check: str
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
        Sets the expert persona and provides detailed instructions, including
        explicit warnings about logical fallacies, few-shot examples, and a
        mandatory, rigorous chain-of-thought process.
        """
        return """
You are a meticulous software architect with deep expertise in non-functional requirements. Your task is to determine if a text snippet from a software project provides concrete evidence of a specific architectural **mechanism** used to achieve a quality attribute.

Your analysis must be rigorous. You are not a keyword spotter. You are a design reviewer.

---
### Core Principle: Mechanism vs. Feature vs. Problem

This is the most critical distinction. You must differentiate between:
1.  **Architectural Mechanism (The "How" - TRUE POSITIVE):** A description of a specific design or implementation choice made to achieve a quality attribute. This is the **solution**.
    -   *Example:* "We implemented a caching layer to reduce latency."
2.  **System Feature (The "What"):** A description of what the software does functionally.
    -   *Example:* "The system can export reports to PDF."
3.  **System Problem (A Failure):** A description of a bug, error, user complaint, or installation issue.
    -   *Example:* "The application crashes when I click the export button."

---
### Three Critical Traps to Avoid (Common Logical Fallacies)

Your predecessor made common mistakes. You must avoid them.

**1. The Functionality-Quality Conflation:**
   - **DO NOT** mistake a description of a feature for a quality attribute mechanism.
   - **Bad Example:** A progress bar is a *usability feature*, not an *availability mechanism*. It tells the user about a download; it doesn't make the download itself more resilient to failure.

**2. The Problem vs. Solution Fallacy:**
   - **DO NOT** confuse a report of a system failure with a description of a mechanism designed to handle that failure. A bug report is evidence of a *lack* of quality, not the presence of a solution.
   - **Bad Example:** A user reporting "OSError: [E050] Can't find model" is describing a **problem**. An **availability mechanism** would be the system automatically falling back to a default model to prevent a crash.

**3. The Tangential Association Fallacy:**
   - **DO NOT** make weak or speculative leaps. The evidence must be direct.
   - **Bad Example:** "Reducing a file's size on disk" is evidence of *storage optimization*. It is **not** direct evidence of *energy efficiency* (reduced CPU cycles) unless the text explicitly makes that causal link.

---
### Illustrative Examples

To calibrate your judgment, study these examples carefully.

**Case 1: Correctly Identified TRUE POSITIVE (Availability)**
- **Content to Analyze:** "Download to temporary file, then copy to cache dir once finished. Otherwise you get corrupt cache entries if the download gets interrupted."
- **Correct Analysis:** This is a **TRUE POSITIVE**. It describes a specific implementation pattern (atomic write via a temp file) explicitly designed to prevent a fault (data corruption). This is a classic availability/resilience mechanism.

**Case 2: Correctly Identified FALSE POSITIVE (Availability)**
- **Content to Analyze:** "OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory."
- **Correct Analysis:** This is a **FALSE POSITIVE**. It is a user reporting a **problem**—a system failure. It does not describe a mechanism *within the software* designed to handle this failure, such as a failover or a fallback.

---
### Your Response: Mandatory Chain of Thought & Output

You will be given a detailed rubric for the quality attribute. You must follow it strictly. Generate a JSON response by completing the following analysis steps **in order**.

1.  `analysis_problem_vs_solution`: Is the text describing a **solution** (a mechanism implemented by developers) or a **problem** (a bug, user error, crash report)?
2.  `analysis_mechanism_vs_feature`: If it is a solution, does it describe an **architectural mechanism** (how the system achieves a quality) or simply a **functional feature** (what the system does)?
3.  `analysis_causal_link`: Is the link between the mechanism and the quality attribute **direct and explicit** in the text, or is it a **tangential or speculative** association?
4.  `analysis_rubric_check`: Does the described mechanism match the **Inclusion Criteria** and avoid the **Exclusion Criteria** provided in the rubric? (Briefly state how it matches or fails).
5.  `true_positive`: `true` or `false`. This decision must be the logical conclusion of the preceding analysis steps.
6.  `reasoning`: A final, concise summary of your decision, referencing the fallacies or rubric if applicable.
"""

    @classmethod
    def to_prompt(cls, x: pd.Series) -> str:
        """
        Provides the specific, dynamic data for the LLM to analyze, including
        the detailed, structured rubric.
        """
        return f"""
### Data for Evaluation

**1. Quality Attribute:** {x['qa']}

**2. Detailed Rubric:**
{x['qa_rubric']}

**3. Content to Analyze:**
"{x['sentence']}"

Now, apply the analysis steps defined in your system prompt to the data provided above.
"""

    @classmethod
    def filter_and_transform_df_before_processing(cls, df):
        """
        Builds a detailed, structured rubric for each quality attribute based on
        the recommendations in the analysis report. This includes explicit
        inclusion and exclusion criteria.
        """
        qa_details = {"availability": {
            "desc": "Mechanisms that ensure a system remains operational and ready to perform its tasks despite the presence of faults (e.g., hardware failures, network interruptions, software bugs).",
            "inclusion_criteria": [
                "Redundancy/Replication: Descriptions of running multiple instances of a component or service.",
                "Failover: Logic that automatically switches from a failed component to a standby one.",
                "Health Checks & Self-Healing: Processes that monitor component health and automatically restart or replace failed instances.",
                "Caching for Resilience: Using a cache to serve data when the primary data source is unavailable.",
                "Fault Prevention (Data Integrity): Mechanisms designed to prevent data corruption that would cause an outage (e.g., atomic writes)."],
            "exclusion_criteria": [
                "User Installation/Configuration Errors: Reports of `pip install` failing, missing files, or incorrect environment setup.",
                "Requests for Support/Documentation: Questions about how to use a feature.",
                "Functional Bugs: Errors where the system runs but produces an incorrect result.",
                "General Maintenance: Discussions of upgrading versions unless the upgrade itself introduces a specific availability mechanism."]},
            "deployability": {
                "desc": "Mechanisms that automate or simplify the ease, speed, and reliability with which a new version of a system can be delivered to and installed by its users.",
                "inclusion_criteria": ["Mentions of package managers (pip, conda, mamba).",
                                       "Containerization technologies (Dockerfile, docker-compose).",
                                       "Build automation scripts (makefiles, shell scripts for release).",
                                       "CI/CD pipeline configurations (e.g., GitHub Actions workflows).",
                                       "Documentation providing structured guidance for installation across different environments."],
                "exclusion_criteria": ["General discussions of software version numbers.",
                                       "Bug fixes that do not touch upon the release or installation process itself."]},
            "energy efficiency": {
                "desc": "Mechanisms specifically intended to minimize the consumption of operational computing resources, such as CPU cycles, memory, I/O, and electrical power.",
                "inclusion_criteria": [
                    "Algorithmic Optimization: Replacing an algorithm with a less computationally complex one (e.g., 'replaced bubble sort with quicksort to reduce CPU usage').",
                    "Caching/Memoization: Storing the results of expensive computations to avoid re-calculating them.",
                    "Resource Throttling/Power-Saving: Features that reduce resource usage during idle periods or allow for performance trade-offs.",
                    "Minimizing Memory Footprint: Techniques to reduce the amount of RAM used during operation (e.g., 'switched to float16 to halve memory usage')."],
                "exclusion_criteria": ["Storage Size Reduction: Decreasing the size of files on disk.",
                                       "Improved Download/Load Times: Making the program start or install faster.",
                                       "Vague Claims: General, unsubstantiated statements like 'this is more efficient' without specifying the resource being saved."]},
            # NOTE: Rubrics for other QAs can be added here following the same pattern.
            # For now, we will create a default, less detailed rubric for others.
        }

        default_desc = {
            "integrability": "Integrability refers to the ease with which software components or distinct systems can be combined and made to work together effectively as a coherent whole, often supported by mechanisms that reduce coupling and manage dependencies.",
            "interoperability": "Interoperability is the degree to which two or more systems can usefully exchange and correctly interpret meaningful information via their interfaces within a particular context.",
            "modifiability": "Modifiability refers to the ease with which changes, such as adding, deleting, or modifying functionality, quality attributes, capacity, or technology, can be made to a system, ideally involving the fewest distinct elements.",
            "performance": "Performance is a system's ability to meet its timing requirements, encompassing its time-based response to events and its efficiency in resource usage under specified conditions.",
            "reliability": "Reliability describes the degree to which a system, product, or component performs its specified functions under defined conditions for a given period, often closely related to the broader concept of availability.",
            "safety": "Safety refers to the software's ability to avoid entering hazardous states that could cause damage, injury, or loss of life, and to recover or limit harm if such states are entered.",
            "security": "Security is the degree to which a system protects information and data from unauthorised access or manipulation, ensuring confidentiality, integrity, and availability for legitimate users.",
            "testability": "Testability refers to the ease with which software can be made to quickly reveal its faults through execution-based testing, by providing controllability and observability of its state and limiting complexity.",
            "usability": "Usability is concerned with how easily users can accomplish desired tasks and the kind of user support the system provides to facilitate their effectiveness, efficiency, and satisfaction.", }

        def format_rubric(qa_name):
            details = qa_details.get(qa_name)
            if details:
                inclusion_str = "\n".join(f"- {item}" for item in details["inclusion_criteria"])
                exclusion_str = "\n".join(f"- {item}" for item in details["exclusion_criteria"])
                return f"""
**Definition:** {details['desc']}
**Inclusion Criteria (Must describe one of these):**
{inclusion_str}
**Exclusion Criteria (Must NOT be one of these):**
{exclusion_str}
"""
            else:
                return f"**Definition:** {default_desc.get(qa_name, 'No description available.')}"

        df["qa_rubric"] = df["qa"].apply(format_rubric)
        return df

    @classmethod
    def transform_df_before_saving(cls, df):
        return df.drop(columns=["qa_rubric"])


def main():
    QARelevanceCheckStage_v2(hostname=LLMHost.SERVER).execute(
        ["root-project.root.v6-32-06.code_comment.", "root-project.root.v6-32-06.docs.",
         "root-project.root.v6-32-06.issue."], reverse=False)


if __name__ == "__main__":
    main()
