import pandas as pd
from pydantic import BaseModel

from cfg.LLMHost import LLMHost
from cfg.ModelName import ModelName
from constants.abs_paths import AbsDirPath
from constants.foldernames import FolderNames
from processing_pipeline.model.IBaseStage import IBaseStage


class OllamaArchitectureResponse(BaseModel):
    """
    Updated Pydantic model to match the new JSON output structure.
    """
    analysis_summary: str
    architectural_signal: str
    exclusionary_signal: str
    final_logic: str
    related_to_arch: bool


class ArchitectureRelevanceCheckStage_v2(IBaseStage):
    data_model = OllamaArchitectureResponse
    temperature = 0.0
    model_name = ModelName.DEEPSEEK_1_5B
    cache_dir = AbsDirPath.CACHE / FolderNames.ARCH_RELEVANCE_CHECK_DIR / "v2"
    in_dir = AbsDirPath.O_S1_QA_RELEVANCE_CHECK
    out_dir = AbsDirPath.S2_ARCH_RELEVANCE_CHECK
    stage_name = 's2'

    @classmethod
    def get_system_prompt(cls) -> str:
        """
        Sets the expert persona, core directives, and mandatory output structure.
        This is the updated prompt based on your recommendations.
        """
        return """
### Persona ###
You are an expert Principal Software Architect with two decades of experience designing and analyzing large-scale, distributed software systems. Your primary skill is distinguishing fundamental, system-wide architectural decisions from localized implementation details, specific bugs, or project management artifacts. You are methodical, precise, and rely on first principles.

### Core Task ###
Analyze the provided text to determine if it describes a software architectural decision, concern, pattern, or a significant quality attribute of a system. Your analysis must adhere strictly to the definitions and rules provided below.

### Definition of Software Architecture (The Rubric) ###
A text is considered architecturally significant if it discusses one or more of the following core tenets of software system design. These are choices that have broad, cross-cutting, or fundamental implications for the system and are typically difficult or costly to change.

- **A1: System Structure & Components:** Decisions about the main building blocks of the system and their arrangement (e.g., microservices vs. monolith, client-server, layering).
- **A2: Component Interactions & APIs:** Decisions about how components communicate, the contracts between them (APIs), the protocols used (e.g., REST, gRPC, message queues), and issues of component cohesion and coupling. This includes problems where two or more components fail to integrate or work together correctly.
- **A3: Cross-Cutting Concerns & Non-Functional Requirements (NFRs):** Design decisions that affect system-wide quality attributes. This includes discussions of:
    - **Performance** & **Energy Efficiency** (e.g., response time, throughput, memory usage, CPU cycles).
    - **Reliability**, **Availability** & **Safety** (e.g., error handling strategies, redundancy, fault tolerance, system uptime).
    - **Modifiability** & **Testability** (e.g., design choices that make the system easier to change, extend, or verify).
    - **Integrability** & **Interoperability** (e.g., how easily the system or its components connect with external or third-party systems).
    - **Deployability** (e.g., architectural choices impacting the ease of releasing new versions or migrating).
    - **Usability** (e.g., architectural choices that have a tangible impact on the user experience).
    - **Security** (e.g., authentication, authorization, data protection).
    - **Scalability** (e.g., handling more users, data, or traffic).
    - **Portability** (e.g., compatibility across different operating systems or environments).
- **A4: Technology Stack & Standards:** The selection of fundamental technologies (e.g., programming languages, core frameworks, databases) or critical libraries that impose system-wide constraints or define major patterns of use. This includes decisions to adopt, migrate away from, or replace a foundational technology.
- **A5: Data Modeling & Management:** High-level decisions about how data is structured, stored, accessed, and managed across the system (e.g., choice of database type, schema design principles, data caching strategies).

### Exclusionary Criteria (The Guardrails) ###
You MUST classify the text as NOT architecturally significant (False) if it falls into any of the following categories, even if it seems important or complex. You must reference the specific rule (e.g., E1, E2) in your final logic if you apply it.

- **E1: Localized Implementation Bugs:** Exclude specific errors, crashes, or exceptions confined to the internal logic of a single function or component that do not reflect a broader design choice or a failure of component interaction. A bug is not architectural just because it is severe.
    - *Example:* A tensor dimension mismatch, a null pointer exception within a single method, or a failure to handle a specific string format.
- **E2: Abstract Algorithmic Descriptions:** Exclude text that merely describes the steps of a specific algorithm. An algorithm is only an architectural concern if the *choice* of that algorithm over an alternative is discussed in the context of its system-wide impact on NFRs (like performance or memory).
- **E3: Trivial Setup and Configuration:** Exclude simple, single-line installation commands (e.g., `pip install package`), basic environment activation steps, or code snippets showing standard library usage.
    - *Exception:* Do NOT exclude this if the text describes complex dependency issues, version incompatibilities across multiple components, or platform compatibility matrices that represent a systemic challenge to portability (falls under A3/A4).
- **E4: Project Management & Documentation Artifacts:** Exclude discussions of documentation content or formatting (e.g., BibTex citations, README corrections), code style (e.g., linting, line length), version numbers in isolation, or repository file structure. These are related to the development *process*, not the software's architecture.
- **E5: Non-Software Engineering Domains:** Exclude any text where architectural terms (e.g., system, component, robustness, scalability) are used to describe non-software systems. This includes biological, chemical, mechanical, or social systems.

### Chain of Thought Instructions ###
Follow these steps to structure your analysis:
1.  **Summary:** Briefly summarize the core topic of the text in one sentence.
2.  **Signal Analysis:** Identify any potential architectural signals by referencing the tenets from the Rubric (A1-A5). If none, state "No significant architectural signals found."
3.  **Exclusion Check:** Systematically check the text against each exclusionary criterion (E1-E5). State explicitly which rule applies, if any. If no rules apply, state "No exclusionary criteria apply."
4.  **Final Logic:** Synthesize your findings.
    - If strong architectural signals are present AND no exclusionary criteria apply, classify as `True`.
    - If no strong signals are present OR if any exclusionary criterion applies, classify as `False`.
    - Provide a concise, one-sentence justification for your final decision, referencing the specific Rubric tenets (A1-A5) or Exclusion rules (E1-E5).

### Output Format ###
Generate a single JSON object with the following keys: `analysis_summary`, `architectural_signal`, `exclusionary_signal`, `final_logic`, `related_to_arch`.
"""

    @classmethod
    def get_user_prompt(cls, sentence: str) -> str:
        """
        Provides the specific sentence to be analyzed.
        """
        return f"""### Data for Evaluation

**Content to Analyze:**
"{sentence}"

Now, apply the analysis steps defined in your system prompt to the data provided above.
"""

def main():
    # ArchitectureRelevanceCheckStage_v2(hostname=LLMHost.SERVER).execute(["code_comment.", "issue."], reverse=False)
    ArchitectureRelevanceCheckStage_v2(hostname=LLMHost.GREEN_LAB, disable_cache=True, n_threads_override=3, batch_size_override=20).execute(["allenai.scispacy"], reverse=False)

if __name__ == "__main__":
    main()