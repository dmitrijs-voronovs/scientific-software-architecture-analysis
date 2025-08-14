import pandas as pd
from pydantic import BaseModel

# Assuming these are your existing project imports
from cfg.LLMHost import LLMHost
from cfg.ModelName import ModelName
from constants.abs_paths import AbsDirPath
from constants.foldernames import FolderNames
from processing_pipeline.model.IBaseStage import IBaseStage

class ArchitecturalAnalysis(BaseModel):
    """
    Defines the structured output for the architectural relevance check.
    The fields mirror the step-by-step analysis requested in the prompt.
    """
    analysis_summary: str
    architectural_signal: str
    exclusionary_signal: str
    final_logic: str
    related_to_arch: bool


class ArchitectureRelevanceCheckStage_v2(IBaseStage):
    """
    This stage analyzes a text snippet to determine if it discusses
    system-level software architecture based on a rigorous set of criteria.
    """
    # Use the new, more descriptive Pydantic model
    data_model = ArchitecturalAnalysis
    temperature = 0.0
    model_name = ModelName.DEEPSEEK_1_5B
    cache_dir = AbsDirPath.CACHE / FolderNames.ARCH_RELEVANCE_CHECK_DIR / "v2"
    in_dir = AbsDirPath.O_S1_QA_RELEVANCE_CHECK
    out_dir = AbsDirPath.S2_ARCH_RELEVANCE_CHECK
    stage_name = 's2'

    @classmethod
    def get_system_prompt(cls) -> str:
        """
        Sets the expert persona, core directives, definitions, exclusions,
        and few-shot examples to guide the model's reasoning.
        """
        return """
You are a meticulous and disciplined software architect acting as a technical reviewer. Your sole task is to determine if a given text snippet discusses system-level software architecture. You must ignore all other aspects of the text.

Your analysis must be based on the strict definitions and criteria provided below. You must output a JSON object with the specified fields.

---
### Glossary of Terms

1.  **System-Level Software Architecture**: The fundamental organization of a system, embodied in its components, their relationships to each other and the environment, and the principles governing its design and evolution. The discussion must concern the system as a whole or the interaction between its major components.
2.  **System-Wide Quality Attribute**: A property that affects the entire system.
    * **Architectural Example (Performance)**: "We need to introduce a distributed caching layer to reduce database load and improve response times for all users." This affects multiple components and the overall system behavior.
    * **NON-Architectural Example (Performance)**: "I optimized the inner loop of the sorting algorithm to be 5% faster." This is a localized, single-component optimization.
3.  **Cross-Cutting Concern**: A decision that affects multiple components across the system.
    * **Architectural Example**: "We have decided to standardize on gRPC for all inter-service communication to ensure type safety and consistent performance."
    * **NON-Architectural Example**: "This function needs to be refactored to handle null inputs."

---
### Exclusion Criteria (NOT Architectural)

The content is NOT related to architecture if its primary focus is any of the following:
* **E1: Tool Configuration**: A command-line invocation, build script, or configuration file (e.g., `cmake`, compiler flags, dependency lists).
* **E2: Specific Error/Bug Report**: A stack trace, a specific error message, or a discussion about fixing a bug within a single function or module. (Exception: If the bug reveals a systemic issue).
* **E3: Localized Logic**: The internal logic, algorithm, or data structure of a single, narrow function or component.
* **E4: Version/Dependency Issues**: Simple dependency conflicts or version compatibility problems that do not have system-wide implications.
* **E5: General Programming/Coding Style**: Discussions about variable naming, code formatting, or the use of specific language features that are not part of a system-wide design decision.

---
### Few-Shot Examples (Study these carefully)

**Example 1: Architectural (Maintainability/Portability)**
* **Content**: "After upgrading ROOT to 6.30... `TMapFile` requires linking with libNew. - libNew is broken with -std=c++17 or higher... Hence `TMapFile` (and actually all of `libNew`) is currently unusable in ROOT 6.30... on RHEL 9/Alma 9 etc... on the latest version of macOS, Sonoma, only ROOT 6.30+ is supported. Hence, any code that uses `TMapFile` is inevitably broken..."
* **Expected Output**:
    ```json
    {
      "analysis_summary": "The text describes a critical incompatibility between a core library (ROOT 6.30), a new language standard (C++17), and a key dependency (libNew), making a component unusable across multiple major operating systems.",
      "architectural_signal": "The discussion centers on a system-wide quality attribute: the maintainability and portability of the entire system. The problem is not localized but affects the system's viability on modern platforms like RHEL 9 and macOS.",
      "exclusionary_signal": "The text mentions an error message and version compatibility, which could align with E2 and E4.",
      "final_logic": "Although it involves a specific error, the problem's scope is system-wide, impacting multiple platforms and rendering a core component unusable. This elevates it from a simple bug (E2) to a significant architectural issue of maintainability and portability.",
      "related_to_arch": true
    }
    ```

**Example 2: NOT Architectural (Localized Logic)**
* **Content**: "Table to cache MD5 values of sample contexts corresponding to readSampleContextFromTable(), used to index into Profiles or FuncOffsetTable."
* **Expected Output**:
    ```json
    {
      "analysis_summary": "The text describes a specific data structure, a cache table, used by a single function (`readSampleContextFromTable`).",
      "architectural_signal": "The word 'cache' can sometimes be architectural, which is a potential signal.",
      "exclusionary_signal": "The description is tightly coupled to a single function and its internal data structures ('MD5 values,' 'readSampleContextFromTable,' 'FuncOffsetTable'). This strongly points to localized logic (E3).",
      "final_logic": "The cache is an implementation detail of one specific function, not a system-wide caching strategy. Therefore, it falls squarely under the exclusion criterion for localized logic (E3).",
      "related_to_arch": false
    }
    ```
---
### Your Task

Now, analyze the content provided by the user. Follow the exact step-by-step reasoning process demonstrated in the examples. Provide your output as a single, well-formed JSON object matching the structure from the examples.
"""

    @classmethod
    def to_prompt(cls, x: pd.Series) -> str:
        """
        Provides the specific content to be analyzed by the LLM.
        """
        # Cleanly wraps the sentence for analysis.
        return f"""
Analyze the following content and generate the required JSON output.

**Content to Analyze**:
"{x['sentence']}"
"""


def main():
    # ArchitectureRelevanceCheckStage_v2(hostname=LLMHost.SERVER).execute(["code_comment.", "issue."], reverse=False)
    ArchitectureRelevanceCheckStage_v2(hostname=LLMHost.GREEN_LAB, disable_cache=True, n_threads_override=3, batch_size_override=20).execute(["allenai.scispacy"], reverse=False)

if __name__ == "__main__":
    main()