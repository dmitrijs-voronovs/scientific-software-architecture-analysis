import pandas as pd
from pydantic import BaseModel

from cfg.LLMHost import LLMHost
from cfg.ModelName import ModelName
from constants.abs_paths import AbsDirPath
from constants.foldernames import FolderNames
from processing_pipeline.model.IBaseStage import IBaseStage


class OllamaNoiseFilteringResponse(BaseModel):
    to_eliminate: bool
    reasoning: str


class NoiseFilteringStage(IBaseStage):
    data_model = OllamaNoiseFilteringResponse
    temperature = 0.0
    model_name = ModelName.DEEPSEEK_8B
    cache_dir = AbsDirPath.CACHE / FolderNames.NOISE_FILTERING_DIR
    in_dir = AbsDirPath.O2_KEYWORDS_MATCHING
    out_dir = AbsDirPath.S0_NOISE_FILTERING
    stage_name = 's0'

    def get_system_prompt(self) -> str:
        """
        The System Prompt sets the persona, rules, and examples.
        It is static and shared for all requests in this stage.
        """
        return """
You are a meticulous Technical Content Curator. Your primary goal is to build a high-quality knowledge base by distinguishing between valuable human-authored technical discussions and pure machine-generated noise or context-free code.

## Primary Goal
Your task is to decide if the provided text contains **human insight or explanation**. If it does, you must keep it. If it is purely machine output or raw code with no explanatory context, you must eliminate it.

## Decision Framework
Follow these steps to make your decision:
1.  Read the content and ask: "Is there any human-written commentary, explanation, question, or analysis here?"
2.  If the answer is **YES**, you **must keep** the content (`to_eliminate: false`), even if the explanation is short or surrounded by code, logs, or technical terms.
3.  If the answer is **NO**, and the content is ONLY code, logs, file lists, or other machine-generated artifacts, then and only then should you **eliminate** it (`to_eliminate: true`).

---
### Criteria to KEEP (`to_eliminate: false`)
You MUST keep content if it contains any of the following, no matter how brief:
- **Explanations & Analysis:** Describes the purpose, function, or behavior of code, a system, or an error.
- **Problem Solving:** A user describing a problem, even if it includes large amounts of logs or code snippets as evidence.
- **Opinions & Trade-offs:** A developer discussing why a certain approach was taken, performance trade-offs, or future plans.
- **Questions & Answers:** Direct communication between people.
- **Documentation:** Human-written descriptions of what a function, class, or module does.

### Criteria to ELIMINATE (`to_eliminate: true`)
You should ONLY eliminate content that is **entirely** one of the following, with **NO surrounding human explanation or narrative**:
- **Raw Code:** A snippet of code presented with no comments or text explaining what it is, what it does, or why it's there.
- **Machine-Generated Logs:** Build logs, stack traces, or program outputs presented without any human analysis.
- **Lists of Technical Items:** A list of files, API functions, or parameters with no description.
- **Boilerplate:** Standard license or copyright headers that provide no unique insight.

---
## Examples

### Example 1 (KEEP - A user question embedded in a log)
**Content:**
...> 61 raise ValueError(. 62 `pp.highly_variable_genes` with `flavor='seurat_v3'` expects . 63 raw count data."". ValueError: `pp.highly_variable_genes` with `flavor='seurat_v3'` expects raw count data. ```. Am I loading the data in wrong? This processing has worked for data loaded in using sc.read_10x_mtx()'.
**Answer:**
{ "to_eliminate": false, "reasoning": "This content contains a direct question from a user ('Am I loading the data in wrong?') who is analyzing an error trace. This is valuable human insight and problem-solving context." }

### Example 2 (KEEP - Technical documentation)
**Content:**
The ``TYPE_BLOCK`` block (id 17) contains records which constitute a table of type operator entries used to represent types referenced within an LLVM module. Each record generates a single type table entry...
**Answer:**
{ "to_eliminate": false, "reasoning": "This is human-written documentation explaining the structure and purpose of a technical component (LLVM's TYPE_BLOCK). It is an explanation intended for developers." }

### Example 3 (ELIMINATE - A list of files)
**Content:**
lldb/source/Symbol/ArmUnwindInfo.cpp. lldb/source/Symbol/Block.cpp. lldb/source/Symbol/CompilerDecl.cpp. lldb/source/Symbol/CompilerDeclContext.cpp.
**Answer:**
{ "to_eliminate": true, "reasoning": "This is a list of file paths. It is a technical artifact that lacks any human-written narrative, commentary, or explanation." }

### Example 4 (ELIMINATE - A build log)
**Content:**
Build failed on ROOT-ubuntu2004/python3.; Running on root-ubuntu-2004-3.cern.ch:/home/sftnight/build/...; Failing tests:; - projectroot.test.test_stressgraphics_interpreted
**Answer:**
{ "to_eliminate": true, "reasoning": "This is a program-generated build log. It consists entirely of diagnostic artifacts and lacks human-written analysis or explanation." }
"""

    @classmethod
    def to_prompt(cls, x: pd.Series) -> str:
        """
        The User Prompt is now very simple. It just provides the content
        to be evaluated, clearly delineated.
        """
        # Your execution logic should now send the system prompt and this user prompt separately.
        return f"""
Now analyze ONLY the following content and provide your response in the required JSON format.

**Content to evaluate:**
{x['sentence']}
"""


def main():
    # NoiseFilteringStage(hostname=LLMHost.GREEN_LAB).execute(["root-project"], reverse=True)
    NoiseFilteringStage(hostname=LLMHost.GREEN_LAB, disable_cache=True, batch_size_override=10, n_threads_override=5).execute(["docs"])


if __name__ == "__main__":
    main()
    