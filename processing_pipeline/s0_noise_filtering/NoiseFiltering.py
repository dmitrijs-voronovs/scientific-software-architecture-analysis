import pandas as pd
from pydantic import BaseModel

from cfg.LLMHost import LLMHost
from cfg.ModelName import ModelName
from constants.abs_paths import AbsDirPath
from constants.foldernames import FolderNames
from processing_pipeline.model.BaseStage import BaseStage


class OllamaFormatValidityResponse(BaseModel):
    to_eliminate: bool
    reasoning: str


class NoiseFilteringStage(BaseStage):
    data_model = OllamaFormatValidityResponse
    temperature = 0.0
    model_name = ModelName.DEEPSEEK_8B
    cache_dir = AbsDirPath.CACHE / FolderNames.NOISE_FILTERING_DIR
    in_dir = AbsDirPath.O2_KEYWORDS_MATCHING
    out_dir = AbsDirPath.S0_NOISE_FILTERING
    stage_name = 's0'

    @classmethod
    def to_prompt(cls, x: pd.Series) -> str:
        return f"""
You are an expert in analyzing and categorizing text content. Your task is to evaluate whether the given **target content** should be filtered out. The goal is to identify and **keep** content that consists of meaningful human-written prose, explanation, or analysis intended for human readers, and to **filter out** content that is primarily non-prose programmatic or technical artifacts.

## Instructions:
For each input, return:
1. `to_eliminate`: true or false — should this content be eliminated?
2. `reasoning`: Brief explanation of why the decision was made.

### Keep Content That:
- Is written for human readers and contains **significant natural language, explanation, commentary, analysis, or discussion**.
- Reflects **communication intended for developers or users**, such as thoughtful suggestions, critiques, or explanations of implementation strategies and trade-offs. This includes bug reports and detailed commit messages.
- Includes **scientific, academic, or detailed technical discussions**, even if highly formal or specialized.
- **Crucially:** This content should be kept **even if it is embedded within or formatted as** technical artifacts (like code comments, log snippets, or documentation with tags). The key is the **substance and intent** of the natural language text. The ratio of prose to code is not the determining factor.

### Eliminate Content That:
- Is **primarily** composed of non-prose programmatic or technical artifacts, **lacking a significant natural language narrative, explanation, or discussion**.
- Consists mainly of:
 - **Pure executable code or formal syntax** (e.g., function bodies without comments, simple variable declarations) without an explanatory narrative.
 - **Program output, logs, or error traces** presented without any surrounding human analysis or interpretation.
 - **Formal configuration, data structures, or build specifications lacking explanatory comments/text** (e.g., pure YAML/JSON data, simple Makefile rules).
 - **Version control metadata lacking explanatory commit messages** (e.g., diff hunks, file path changes).
 - **Formal API signatures or technical interface definitions** without accompanying prose that explains their purpose and usage in a descriptive way.
 - **Boilerplate text** that does not provide any unique insight or explanation, such as standard license or copyright headers.

## Examples (for reference only – do not analyze):

### Example 1 (Eliminate)
**Content:**
Build failed on ROOT-ubuntu2004/python3.; Running on root-ubuntu-2004-3.cern.ch:/home/sftnight/build/...; Failing tests:; - projectroot.test.test_stressgraphics_interpreted
**Answer:**
to_eliminate: true
reasoning: This is a program-generated build log. It consists entirely of diagnostic artifacts and lacks human-written analysis or explanation.

### Example 2 (Eliminate)
**Content:**
scanpy.neighbors.connectivities. scanpy.neighbors.distances. scanpy.neighbors.eigen_basis.
**Answer:**
to_eliminate: true
reasoning: This is a list of API functions. It is a technical artifact and lacks any human-written narrative, commentary, or explanation.

### Example 3 (Keep)
**Content:**
I tested the new parallelization strategy. Simulation time dropped 30%, but memory usage increased. We may need more efficient data structures.
**Answer:**
to_eliminate: false
reasoning: This is a natural language explanation of performance trade-offs and a suggestion for future work. It is a clear example of human-to-human communication.

### Example 4 (Keep)
**Content:**
[RF] Wrong integral value, possibly problems with some global caching. It seems `ws.pdf(""signal"")->plotOn(frame);` creates some global cache of the integral which is used with newly created objects, causing the issue.
**Answer:**
to_eliminate: false
reasoning: This is a bug report that describes a technical problem and analyzes the potential cause (`global caching` from a specific function call). This is valuable human-written analysis, not just a log.

### Example 5 (Keep)
**Content:**
@name Mutation APIs
These methods provide APIs for submitting updates to the DominatorTree. Note: There are two strategies, Eager and Lazy. It is recommended to use the Lazy strategy for multiple updates to improve performance.
**Answer:**
to_eliminate: false
reasoning: This content, despite using documentation tags like `@name`, provides a detailed explanation of two different technical strategies and gives a recommendation. The substance is a human-written explanation of design and trade-offs.

### Example 6 (Eliminate)
**Content:**
History:
Version 17.8, add new method GetDXDY()
Version 17.7, updates in the TUnfold implementation
Version 17.5, fix memory leak
**Answer:**
to_eliminate: true
reasoning: This is a structured changelog that is simply a list of changes. It lacks a narrative or detailed explanation of the *reasons* for the changes, making it a low-value technical artifact.

---

## Now analyze ONLY the following content:

**Content to evaluate:**
{x['sentence']}
"""

def main():
    NoiseFilteringStage(hostname=LLMHost.GREEN_LAB).execute(["root-project"], reverse=False)


if __name__ == "__main__":
    main()
    