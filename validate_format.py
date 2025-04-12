import json
import math
import os
import re
import shelve
import signal
import sys
from pathlib import Path
from typing import List

import dotenv
import pandas as pd
import requests
from langchain_ollama import ChatOllama
from loguru import logger
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, RetryError, wait_fixed
from tqdm import tqdm

from constants.foldernames import FolderNames
from extract_quality_attribs_from_docs import MatchSource
from metadata.repo_info.repo_info import credential_list
from utils.utils import create_logger_path


# Load environment variables from .env file
dotenv.load_dotenv()


class OllamaFormatValidityResponse(BaseModel):
    to_eliminate: bool
    reason: str

# @retry(stop=stop_after_attempt(6), wait=wait_fixed(3), after=lambda retry_state: logger.warning(retry_state),
#     reraise=True, )
def request_ollama_chain(prompts: List[str], base_url: str) -> List[OllamaFormatValidityResponse]:
    # model_name = "gemma"
    # model_name = "gemma2"
    model_name = "deepseek-r1:8b"
    model  = ChatOllama(model=model_name, temperature=0.0, base_url=base_url, format=OllamaFormatValidityResponse.model_json_schema())
    batch_answers = model.batch(prompts)
    return [OllamaFormatValidityResponse.model_validate_json(answer.content) for answer in batch_answers]


to_prompt_v1 = lambda x: \
f"""You are an expert in analyzing and categorizing text content. Your task is to evaluate whether the given text contains meaningful human-readable sentences or if it consists primarily of logs, code samples, or programmatic API description that should be filtered out.

For each input text, analyze it and determine:
1. Whether it should be eliminated (true/false)
2. The reason for elimination (if applicable)

Evaluation criteria:
- Eliminate text that consists primarily of:
  * Code snippets or samples (marked by syntax, keywords like "if/else", brackets, etc.)
  * Program logs or error messages (timestamps, error codes, stack traces)
  * API documentation or specifications (parameter lists, return types)
  * Configuration files or build system output
  * Version control metadata or comments
  * Compiler/interpreter output or warnings
- Keep text that contains:
  * Complete, meaningful sentences in natural language
  * Explanatory or descriptive content
  * Human-written prose discussing concepts or ideas

Content: {x['sentence']}

Examples:

Content: Build failed on ROOT-ubuntu2004/python3.; Running on root-ubuntu-2004-3.cern.ch:/home/sftnight/build/workspace/root-pullrequests-build; [See console output](https://lcgapp-services.cern.ch/root-jenkins/job/root-pullrequests-build/183837/console).; ### Failing tests:; - [projectroot.test.test_stressgraphics_interpreted](https://lcgapp-services.cern.ch/root-jenkins/job/root-pullrequests-build/183837/testReport/projectroot/test/test_stressgraphics_interpreted/)
Answer: to_eliminate: false
Reasoning: The content only consists of logs and no other text.

Content: recision><conversion specifier>`` where:. * ``#`` is an optional flag available for hex values (see; ``<conversion specifier>`` below) which requires the value matched to be; prefixed by ``0x``.; * ``.<precision>`` is an optional printf-style precision specifier in which; ``<precision>`` indicates the minimum number of digits that the value matched; must have, expecting leading zeros if needed. * ``<conversion specifier>`` is an optional scanf-style conversion specifier; to indicate what number format to match (e.g. hex number). Currently; accepted format specifiers are ``%u``, ``%d``, ``%x`` and ``%X``. If absent,; the format specifier defaults to ``%u``. For example:. .. code-block:: llvm. ; CHECK: mov r[[#REG:]], 0x[[#%.8X,ADDR:]]. would match ``mov r5, 0x0000FEFE`` and set ``REG`` to the value ``5`` and; ``ADDR`` to the value ``0xFEFE``. Note that due to the precision it would fail; to match ``mov r5, 0xFEFE``. As a result of the numeric variable definition being optional, it is possible; to only check that a numeric value is present in a given format. This can be; useful when the value itself is not useful, for instance:. .. code-block:: gas. ; CHECK-NOT: mov r0, r[[#]]. to check that a value is synthesized rather than moved around. The syntax of a numeric substitution is; ``[[#%<fmtspec>, <constraint> <expr>]]`` where:. * ``<fmtspec>`` is the same format specifier as for defining a variable but; in this context indicating how a numeric expression value should be matched; against. If absent, both components of the format specifier are inferred from; the matching format of the numeric variable(s) used by the expression; constraint if any, and defaults to ``%u`` if no numeric variable is used,; denoting that the value should be unsigned with no leading zeros. In case of; conflict between format specifiers of several numeric variables, the; conversion specifier becomes mandatory but the precision specifier remains; optional. * ``<constraint>`` is the constraint de
Answer: to_eliminate: false
Reasoning: The content only consists of programmatic interface description and no other text.

Content: I've been testing the latest updates on our scientific computing framework, and I'm impressed by the improvements in performance. The new parallelization strategy has reduced our simulation times by about 30% on our cluster. However, I noticed that memory usage spikes significantly when handling large datasets. Perhaps we could explore more efficient data structures or caching mechanisms to mitigate this issue? @TeamLead, would it be possible to allocate some resources to investigate this further?
Answer: to_eliminate: false
Reasoning: The content contains meaningful human-written sentences in natural language  

Content: I've been experimenting with the GPU acceleration patches in this PR, and the results are promising. We're seeing a significant boost in matrix operations, which is crucial for our machine learning models. However, I encountered some inconsistencies when running on different GPU architectures. It might be beneficial to add more comprehensive tests to ensure compatibility across various hardware configurations. @DevTeam, could we discuss implementing a more robust testing framework for this?
Answer: to_eliminate: false
Reasoning: The content contains meaningful human-written sentences in natural language  

Content: I've been exploring the energy efficiency improvements in our latest scientific computing framework, and I'm excited about the potential savings. By implementing parallelization strategies and optimizing data structures, we've reduced energy consumption by approximately 25% on our cluster. However, I think we could further enhance this by integrating tools like EnergyMeter to monitor and optimize energy usage in real-time. Additionally, exploring autotuning frameworks like ytopt could help balance performance and energy efficiency across different hardware configurations. @DevTeam, would it be feasible to allocate resources for integrating these tools and conducting a comprehensive energy efficiency analysis?
Answer: to_eliminate: false
Reasoning: The content contains meaningful human-written sentences in natural language  
"""

to_prompt_v2 = lambda x: \
f"""
You are an expert in analyzing and categorizing text content. Your task is to evaluate whether the given **target content** should be filtered out, based on whether it consists of meaningful human-written prose or instead mainly contains programmatic or technical artifacts like logs or code.

## Instructions:
For each input, return:
1. `to_eliminate`: true or false — should this content be eliminated?
2. `reasoning`: Brief explanation of why the decision was made.

## Eliminate content that consists primarily of:
- Code snippets or samples (e.g., `if/else`, brackets, keywords)
- Program logs or error messages (timestamps, error codes, stack traces)
- API documentation or technical specifications
- Configuration files or build output
- Version control metadata or system messages
- Compiler/interpreter output

## Keep content that contains:
- Complete, meaningful sentences in natural language
- Explanatory or descriptive content
- Human-written prose discussing concepts, decisions, or ideas

## Examples (for reference only – do not analyze):

### Example 1
**Content:** Build failed on ROOT-ubuntu2004/python3.; Running on root-ubuntu-2004-3.cern.ch:/home/sftnight/build/workspace/root-pullrequests-build; [See console output](https://lcgapp-services.cern.ch/root-jenkins/job/root-pullrequests-build/183837/console).; ### Failing tests:; - [projectroot.test.test_stressgraphics_interpreted](https://lcgapp-services.cern.ch/root-jenkins/job/root-pullrequests-build/183837/testReport/projectroot/test/test_stressgraphics_interpreted/)
**Answer:**
to_eliminate: true
reasoning: The content only consists of logs and no other text.

### Example 2
**Content:** recision><conversion specifier>`` where:. * ``#`` is an optional flag available for hex values (see; ``<conversion specifier>`` below) which requires the value matched to be; prefixed by ``0x``.; * ``.<precision>`` is an optional printf-style precision specifier in which; ``<precision>`` indicates the minimum number of digits that the value matched; must have, expecting leading zeros if needed. * ``<conversion specifier>`` is an optional scanf-style conversion specifier; to indicate what number format to match (e.g. hex number). Currently; accepted format specifiers are ``%u``, ``%d``, ``%x`` and ``%X``. If absent,; the format specifier defaults to ``%u``. For example:. .. code-block:: llvm. ; CHECK: mov r[[#REG:]], 0x[[#%.8X,ADDR:]]. would match ``mov r5, 0x0000FEFE`` and set ``REG`` to the value ``5`` and; ``ADDR`` to the value ``0xFEFE``. Note that due to the precision it would fail; to match ``mov r5, 0xFEFE``. As a result of the numeric variable definition being optional, it is possible; to only check that a numeric value is present in a given format. This can be; useful when the value itself is not useful, for instance:. .. code-block:: gas. ; CHECK-NOT: mov r0, r[[#]]. to check that a value is synthesized rather than moved around. The syntax of a numeric substitution is; ``[[#%<fmtspec>, <constraint> <expr>]]`` where:. * ``<fmtspec>`` is the same format specifier as for defining a variable but; in this context indicating how a numeric expression value should be matched; against. If absent, both components of the format specifier are inferred from; the matching format of the numeric variable(s) used by the expression; constraint if any, and defaults to ``%u`` if no numeric variable is used,; denoting that the value should be unsigned with no leading zeros. In case of; conflict between format specifiers of several numeric variables, the; conversion specifier becomes mandatory but the precision specifier remains; optional. * ``<constraint>`` is the constraint de
**Answer:**
to_eliminate: true
reasoning: The content only consists of programmatic interface description and no other text.

### Example 3
**Content:** I've been testing the latest updates on our scientific computing framework, and I'm impressed by the improvements in performance. The new parallelization strategy has reduced our simulation times by about 30% on our cluster. However, I noticed that memory usage spikes significantly when handling large datasets. Perhaps we could explore more efficient data structures or caching mechanisms to mitigate this issue? @TeamLead, would it be possible to allocate some resources to investigate this further?
**Answer:**
to_eliminate: false
reasoning: The content contains meaningful human-written sentences in natural language

### Example 4
**Content:** I've been experimenting with the GPU acceleration patches in this PR, and the results are promising. We're seeing a significant boost in matrix operations, which is crucial for our machine learning models. However, I encountered some inconsistencies when running on different GPU architectures. It might be beneficial to add more comprehensive tests to ensure compatibility across various hardware configurations. @DevTeam, could we discuss implementing a more robust testing framework for this?
**Answer:**
to_eliminate: false
reasoning: The content contains meaningful human-written sentences in natural language

### Example 5
**Content:** I've been exploring the energy efficiency improvements in our latest scientific computing framework, and I'm excited about the potential savings. By implementing parallelization strategies and optimizing data structures, we've reduced energy consumption by approximately 25% on our cluster. However, I think we could further enhance this by integrating tools like EnergyMeter to monitor and optimize energy usage in real-time. Additionally, exploring autotuning frameworks like ytopt could help balance performance and energy efficiency across different hardware configurations. @DevTeam, would it be feasible to allocate resources for integrating these tools and conducting a comprehensive energy efficiency analysis?
**Answer:**
to_eliminate: false
reasoning: The content contains meaningful human-written sentences in natural language

### Example 6
**Content:** is relatively simple;; they ultimately form a linked list of every clobber that dominates the; ``MemoryAccess`` that you're trying to optimize. In other words, the; ``definingAccess`` of a ``MemoryDef`` is always the nearest dominating; ``MemoryDef`` or ``MemoryPhi`` of said ``MemoryDef``. Use and Def optimization; ------------------------. ``MemoryUse``\ s keep a single operand, which is their defining or optimized; access.; Traditionally ``MemorySSA`` optimized ``MemoryUse``\ s at build-time, up to a; given threshold.; Specifically, the operand of every ``MemoryUse`` was optimized to point to the; actual clobber of said ``MemoryUse``. This can be seen in the above example; the; second ``MemoryUse`` in ``if.end`` has an operand of ``1``, which is a; ``MemoryDef`` from the entry block. This is done to make walking,; value numbering, etc, faster and easier.; As of `this revision <https://reviews.llvm.org/D121381>`_, the default was; changed to not optimize uses at build time, in order to provide the option to; reduce compile-time if the walking is not necessary in a pass. Most users call; the new API ``ensureOptimizedUses()`` to keep the previous behavior and do a; one-time optimization of ``MemoryUse``\ s, if this was not done before.; New pass users are recommended to call ``ensureOptimizedUses()``. Initially it was not possible to optimize ``MemoryDef``\ s in the same way, as we; restricted ``MemorySSA`` to one operand per access.; This was changed and ``MemoryDef``\ s now keep two operands.; The first one, the defining access, is; always the previous ``MemoryDef`` or ``MemoryPhi`` in the same basic block, or; the last one in a dominating predecessor if the current block doesn't have any; other accesses writing to memory. This is needed for walking Def chains.; The second operand is the optimized access, if there was a previous call on the; walker's ``getClobberingMemoryAccess(MA)``. This API will cache information; as part of ``MA``.; Optimizing all ``MemoryDef``
**Answer:**
to_eliminate: false
reasoning: The content contains meaningful human-written sentences in natural language discussing testing experiences
and performance improvements.

---

## Now analyze ONLY the following content:

**Content to evaluate:**  
{x['sentence']}
"""

to_prompt_v3 = lambda x: \
f"""
You are an expert in analyzing and categorizing text content. Your task is to evaluate whether the given **target content** should be filtered out, based on whether it consists of meaningful human-written prose or instead mainly contains programmatic or technical artifacts like logs or code.

## Instructions:
For each input, return:
1. `to_eliminate`: true or false — should this content be eliminated?
2. `reasoning`: Brief explanation of why the decision was made.

## Eliminate content that consists primarily of:
- Code snippets or samples (e.g., `if/else`, brackets, keywords)
- Program logs or error messages (timestamps, error codes, stack traces)
- API documentation or technical specifications
- Configuration files or build output
- Version control metadata or system messages
- Compiler/interpreter output

## Keep content that:
- Does not primarily match any of the elimination criteria
- Serves an informative, descriptive, or communicative purpose
- Appears to be intended for human understanding, regardless of subject matter or style

## Examples (for reference only – do not analyze):

### Example 1
**Content:** Build failed on ROOT-ubuntu2004/python3.; Running on root-ubuntu-2004-3.cern.ch:/home/sftnight/build/workspace/root-pullrequests-build; [See console output](https://lcgapp-services.cern.ch/root-jenkins/job/root-pullrequests-build/183837/console).; ### Failing tests:; - [projectroot.test.test_stressgraphics_interpreted](https://lcgapp-services.cern.ch/root-jenkins/job/root-pullrequests-build/183837/testReport/projectroot/test/test_stressgraphics_interpreted/)
**Answer:**
to_eliminate: true
reasoning: The content only consists of logs and no other text.

### Example 2
**Content:** recision><conversion specifier>`` where:. * ``#`` is an optional flag available for hex values (see; ``<conversion specifier>`` below) which requires the value matched to be; prefixed by ``0x``.; * ``.<precision>`` is an optional printf-style precision specifier in which; ``<precision>`` indicates the minimum number of digits that the value matched; must have, expecting leading zeros if needed. * ``<conversion specifier>`` is an optional scanf-style conversion specifier; to indicate what number format to match (e.g. hex number). Currently; accepted format specifiers are ``%u``, ``%d``, ``%x`` and ``%X``. If absent,; the format specifier defaults to ``%u``. For example:. .. code-block:: llvm. ; CHECK: mov r[[#REG:]], 0x[[#%.8X,ADDR:]]. would match ``mov r5, 0x0000FEFE`` and set ``REG`` to the value ``5`` and; ``ADDR`` to the value ``0xFEFE``. Note that due to the precision it would fail; to match ``mov r5, 0xFEFE``. As a result of the numeric variable definition being optional, it is possible; to only check that a numeric value is present in a given format. This can be; useful when the value itself is not useful, for instance:. .. code-block:: gas. ; CHECK-NOT: mov r0, r[[#]]. to check that a value is synthesized rather than moved around. The syntax of a numeric substitution is; ``[[#%<fmtspec>, <constraint> <expr>]]`` where:. * ``<fmtspec>`` is the same format specifier as for defining a variable but; in this context indicating how a numeric expression value should be matched; against. If absent, both components of the format specifier are inferred from; the matching format of the numeric variable(s) used by the expression; constraint if any, and defaults to ``%u`` if no numeric variable is used,; denoting that the value should be unsigned with no leading zeros. In case of; conflict between format specifiers of several numeric variables, the; conversion specifier becomes mandatory but the precision specifier remains; optional. * ``<constraint>`` is the constraint de
**Answer:**
to_eliminate: true
reasoning: The content only consists of programmatic interface description and no other text.

### Example 3
**Content:** I've been testing the latest updates on our scientific computing framework, and I'm impressed by the improvements in performance. The new parallelization strategy has reduced our simulation times by about 30% on our cluster. However, I noticed that memory usage spikes significantly when handling large datasets. Perhaps we could explore more efficient data structures or caching mechanisms to mitigate this issue? @TeamLead, would it be possible to allocate some resources to investigate this further?
**Answer:**
to_eliminate: false
reasoning: The content contains meaningful human-written sentences in natural language

### Example 4
**Content:** I've been experimenting with the GPU acceleration patches in this PR, and the results are promising. We're seeing a significant boost in matrix operations, which is crucial for our machine learning models. However, I encountered some inconsistencies when running on different GPU architectures. It might be beneficial to add more comprehensive tests to ensure compatibility across various hardware configurations. @DevTeam, could we discuss implementing a more robust testing framework for this?
**Answer:**
to_eliminate: false
reasoning: The content contains meaningful human-written sentences in natural language

### Example 5
**Content:** I've been exploring the energy efficiency improvements in our latest scientific computing framework, and I'm excited about the potential savings. By implementing parallelization strategies and optimizing data structures, we've reduced energy consumption by approximately 25% on our cluster. However, I think we could further enhance this by integrating tools like EnergyMeter to monitor and optimize energy usage in real-time. Additionally, exploring autotuning frameworks like ytopt could help balance performance and energy efficiency across different hardware configurations. @DevTeam, would it be feasible to allocate resources for integrating these tools and conducting a comprehensive energy efficiency analysis?
**Answer:**
to_eliminate: false
reasoning: The content contains meaningful human-written sentences in natural language

### Example 6
**Content:** is relatively simple;; they ultimately form a linked list of every clobber that dominates the; ``MemoryAccess`` that you're trying to optimize. In other words, the; ``definingAccess`` of a ``MemoryDef`` is always the nearest dominating; ``MemoryDef`` or ``MemoryPhi`` of said ``MemoryDef``. Use and Def optimization; ------------------------. ``MemoryUse``\ s keep a single operand, which is their defining or optimized; access.; Traditionally ``MemorySSA`` optimized ``MemoryUse``\ s at build-time, up to a; given threshold.; Specifically, the operand of every ``MemoryUse`` was optimized to point to the; actual clobber of said ``MemoryUse``. This can be seen in the above example; the; second ``MemoryUse`` in ``if.end`` has an operand of ``1``, which is a; ``MemoryDef`` from the entry block. This is done to make walking,; value numbering, etc, faster and easier.; As of `this revision <https://reviews.llvm.org/D121381>`_, the default was; changed to not optimize uses at build time, in order to provide the option to; reduce compile-time if the walking is not necessary in a pass. Most users call; the new API ``ensureOptimizedUses()`` to keep the previous behavior and do a; one-time optimization of ``MemoryUse``\ s, if this was not done before.; New pass users are recommended to call ``ensureOptimizedUses()``. Initially it was not possible to optimize ``MemoryDef``\ s in the same way, as we; restricted ``MemorySSA`` to one operand per access.; This was changed and ``MemoryDef``\ s now keep two operands.; The first one, the defining access, is; always the previous ``MemoryDef`` or ``MemoryPhi`` in the same basic block, or; the last one in a dominating predecessor if the current block doesn't have any; other accesses writing to memory. This is needed for walking Def chains.; The second operand is the optimized access, if there was a previous call on the; walker's ``getClobberingMemoryAccess(MA)``. This API will cache information; as part of ``MA``.; Optimizing all ``MemoryDef``
**Answer:**
to_eliminate: false
reasoning: The content contains meaningful human-written sentences in natural language discussing testing experiences
and performance improvements.

---

## Now analyze ONLY the following content:

**Content to evaluate:**  
{x['sentence']}
"""

to_prompt_v4 = lambda x: \
f"""
You are an expert in analyzing and categorizing text content. Your task is to evaluate whether the given **target content** should be filtered out, based on whether it consists of meaningful human-written prose or instead mainly contains programmatic or technical artifacts like logs or code.

## Instructions:
For each input, return:
1. `to_eliminate`: true or false — should this content be eliminated?
2. `reasoning`: Brief explanation of why the decision was made.

### Eliminate content that consists primarily of:
- Code snippets or program structure  
  *(e.g., `if/else`, `for` loops, brackets, language-specific syntax or keywords)*
- Program output, logs, or error traces  
  *(e.g., timestamps, error codes, stack traces, test failures)*
- Configuration files, scripts, or build system output  
  *(e.g., YAML, JSON, shell scripts, Makefiles, CMake output, compiler warnings)*
- Version control metadata or commit messages  
  *(e.g., diffs, git logs, merge info, file paths with change indicators)*
- API documentation or technical interface specifications  
  *(e.g., function signatures, parameter lists, return types, type annotations, code-style formatting like `Args:`, `Returns:`)*

### Keep content that:
- Does **not** fall primarily into any of the elimination categories
- Conveys **insight, explanation, or information** intended for human interpretation
- Includes **expository or analytical discussion**, even if highly specialized, formal, or domain-specific (e.g., scientific, academic, technical, or medical writing)
- May contain structured or technical language, as long as it is **not primarily in the format of code, logs, or machine output**

## Examples (for reference only – do not analyze):

### Example 1
**Content:** Build failed on ROOT-ubuntu2004/python3.; Running on root-ubuntu-2004-3.cern.ch:/home/sftnight/build/workspace/root-pullrequests-build; [See console output](https://lcgapp-services.cern.ch/root-jenkins/job/root-pullrequests-build/183837/console).; ### Failing tests:; - [projectroot.test.test_stressgraphics_interpreted](https://lcgapp-services.cern.ch/root-jenkins/job/root-pullrequests-build/183837/testReport/projectroot/test/test_stressgraphics_interpreted/)
**Answer:**
to_eliminate: true
reasoning: The content only consists of logs and no other text.

### Example 2
**Content:** recision><conversion specifier>`` where:. * ``#`` is an optional flag available for hex values (see; ``<conversion specifier>`` below) which requires the value matched to be; prefixed by ``0x``.; * ``.<precision>`` is an optional printf-style precision specifier in which; ``<precision>`` indicates the minimum number of digits that the value matched; must have, expecting leading zeros if needed. * ``<conversion specifier>`` is an optional scanf-style conversion specifier; to indicate what number format to match (e.g. hex number). Currently; accepted format specifiers are ``%u``, ``%d``, ``%x`` and ``%X``. If absent,; the format specifier defaults to ``%u``. For example:. .. code-block:: llvm. ; CHECK: mov r[[#REG:]], 0x[[#%.8X,ADDR:]]. would match ``mov r5, 0x0000FEFE`` and set ``REG`` to the value ``5`` and; ``ADDR`` to the value ``0xFEFE``. Note that due to the precision it would fail; to match ``mov r5, 0xFEFE``. As a result of the numeric variable definition being optional, it is possible; to only check that a numeric value is present in a given format. This can be; useful when the value itself is not useful, for instance:. .. code-block:: gas. ; CHECK-NOT: mov r0, r[[#]]. to check that a value is synthesized rather than moved around. The syntax of a numeric substitution is; ``[[#%<fmtspec>, <constraint> <expr>]]`` where:. * ``<fmtspec>`` is the same format specifier as for defining a variable but; in this context indicating how a numeric expression value should be matched; against. If absent, both components of the format specifier are inferred from; the matching format of the numeric variable(s) used by the expression; constraint if any, and defaults to ``%u`` if no numeric variable is used,; denoting that the value should be unsigned with no leading zeros. In case of; conflict between format specifiers of several numeric variables, the; conversion specifier becomes mandatory but the precision specifier remains; optional. * ``<constraint>`` is the constraint de
**Answer:**
to_eliminate: true
reasoning: The content only consists of programmatic interface description and no other text.

### Example 3
**Content:** I've been testing the latest updates on our scientific computing framework, and I'm impressed by the improvements in performance. The new parallelization strategy has reduced our simulation times by about 30% on our cluster. However, I noticed that memory usage spikes significantly when handling large datasets. Perhaps we could explore more efficient data structures or caching mechanisms to mitigate this issue? @TeamLead, would it be possible to allocate some resources to investigate this further?
**Answer:**
to_eliminate: false
reasoning: The content contains meaningful human-written sentences in natural language

### Example 4
**Content:** I've been experimenting with the GPU acceleration patches in this PR, and the results are promising. We're seeing a significant boost in matrix operations, which is crucial for our machine learning models. However, I encountered some inconsistencies when running on different GPU architectures. It might be beneficial to add more comprehensive tests to ensure compatibility across various hardware configurations. @DevTeam, could we discuss implementing a more robust testing framework for this?
**Answer:**
to_eliminate: false
reasoning: The content contains meaningful human-written sentences in natural language

### Example 5
**Content:** I've been exploring the energy efficiency improvements in our latest scientific computing framework, and I'm excited about the potential savings. By implementing parallelization strategies and optimizing data structures, we've reduced energy consumption by approximately 25% on our cluster. However, I think we could further enhance this by integrating tools like EnergyMeter to monitor and optimize energy usage in real-time. Additionally, exploring autotuning frameworks like ytopt could help balance performance and energy efficiency across different hardware configurations. @DevTeam, would it be feasible to allocate resources for integrating these tools and conducting a comprehensive energy efficiency analysis?
**Answer:**
to_eliminate: false
reasoning: The content contains meaningful human-written sentences in natural language

### Example 6
**Content:** is relatively simple;; they ultimately form a linked list of every clobber that dominates the; ``MemoryAccess`` that you're trying to optimize. In other words, the; ``definingAccess`` of a ``MemoryDef`` is always the nearest dominating; ``MemoryDef`` or ``MemoryPhi`` of said ``MemoryDef``. Use and Def optimization; ------------------------. ``MemoryUse``\ s keep a single operand, which is their defining or optimized; access.; Traditionally ``MemorySSA`` optimized ``MemoryUse``\ s at build-time, up to a; given threshold.; Specifically, the operand of every ``MemoryUse`` was optimized to point to the; actual clobber of said ``MemoryUse``. This can be seen in the above example; the; second ``MemoryUse`` in ``if.end`` has an operand of ``1``, which is a; ``MemoryDef`` from the entry block. This is done to make walking,; value numbering, etc, faster and easier.; As of `this revision <https://reviews.llvm.org/D121381>`_, the default was; changed to not optimize uses at build time, in order to provide the option to; reduce compile-time if the walking is not necessary in a pass. Most users call; the new API ``ensureOptimizedUses()`` to keep the previous behavior and do a; one-time optimization of ``MemoryUse``\ s, if this was not done before.; New pass users are recommended to call ``ensureOptimizedUses()``. Initially it was not possible to optimize ``MemoryDef``\ s in the same way, as we; restricted ``MemorySSA`` to one operand per access.; This was changed and ``MemoryDef``\ s now keep two operands.; The first one, the defining access, is; always the previous ``MemoryDef`` or ``MemoryPhi`` in the same basic block, or; the last one in a dominating predecessor if the current block doesn't have any; other accesses writing to memory. This is needed for walking Def chains.; The second operand is the optimized access, if there was a previous call on the; walker's ``getClobberingMemoryAccess(MA)``. This API will cache information; as part of ``MA``.; Optimizing all ``MemoryDef``
**Answer:**
to_eliminate: false
reasoning: The content contains meaningful human-written sentences in natural language discussing testing experiences
and performance improvements.

---

## Now analyze ONLY the following content:

**Content to evaluate:**  
{x['sentence']}
"""

to_prompt_v5 = lambda x: \
f"""
You are an expert in analyzing and categorizing text content. Your task is to evaluate whether the given **target content** should be filtered out, based on whether it consists of meaningful human-written prose or instead mainly contains programmatic or technical artifacts like logs or code.

## Instructions:
For each input, return:
1. `to_eliminate`: true or false — should this content be eliminated?
2. `reasoning`: Brief explanation of why the decision was made.


### Eliminate content that is not intended for human interpretation and consists primarily of:
- Code snippets or program structure  
  *(e.g., `if/else`, `for` loops, braces, language-specific syntax or keywords)*
- Program output, logs, or error traces  
  *(e.g., timestamps, error codes, stack traces, unit test results)*
- Configuration files, scripts, or build system output  
  *(e.g., YAML, JSON, Makefiles, shell scripts, compiler output)*
- Version control metadata or commit messages  
  *(e.g., git logs, diffs, merge info, file paths with change indicators)*
- API documentation or technical interface definitions  
  *(e.g., method signatures, parameter tables, annotations, formal docstrings)*

### Keep content that:
- Does **not** fall primarily into any of the elimination categories
- Is written for human readers — including **natural language, explanation, commentary, or analysis**
- Includes **scientific, academic, or technical discussions**, even if highly formal or specialized  
  *(e.g., discussions of model architecture, training benchmarks, research outcomes, biological findings, or engineering analysis)*
- May contain structured or technical vocabulary, **as long as it is not formatted primarily like code or machine output**

## Examples (for reference only – do not analyze):

### Example 1
**Content:** Build failed on ROOT-ubuntu2004/python3.; Running on root-ubuntu-2004-3.cern.ch:/home/sftnight/build/workspace/root-pullrequests-build; [See console output](https://lcgapp-services.cern.ch/root-jenkins/job/root-pullrequests-build/183837/console).; ### Failing tests:; - [projectroot.test.test_stressgraphics_interpreted](https://lcgapp-services.cern.ch/root-jenkins/job/root-pullrequests-build/183837/testReport/projectroot/test/test_stressgraphics_interpreted/)
**Answer:**
to_eliminate: true
reasoning: The content only consists of logs and no other text.

### Example 2
**Content:** recision><conversion specifier>`` where:. * ``#`` is an optional flag available for hex values (see; ``<conversion specifier>`` below) which requires the value matched to be; prefixed by ``0x``.; * ``.<precision>`` is an optional printf-style precision specifier in which; ``<precision>`` indicates the minimum number of digits that the value matched; must have, expecting leading zeros if needed. * ``<conversion specifier>`` is an optional scanf-style conversion specifier; to indicate what number format to match (e.g. hex number). Currently; accepted format specifiers are ``%u``, ``%d``, ``%x`` and ``%X``. If absent,; the format specifier defaults to ``%u``. For example:. .. code-block:: llvm. ; CHECK: mov r[[#REG:]], 0x[[#%.8X,ADDR:]]. would match ``mov r5, 0x0000FEFE`` and set ``REG`` to the value ``5`` and; ``ADDR`` to the value ``0xFEFE``. Note that due to the precision it would fail; to match ``mov r5, 0xFEFE``. As a result of the numeric variable definition being optional, it is possible; to only check that a numeric value is present in a given format. This can be; useful when the value itself is not useful, for instance:. .. code-block:: gas. ; CHECK-NOT: mov r0, r[[#]]. to check that a value is synthesized rather than moved around. The syntax of a numeric substitution is; ``[[#%<fmtspec>, <constraint> <expr>]]`` where:. * ``<fmtspec>`` is the same format specifier as for defining a variable but; in this context indicating how a numeric expression value should be matched; against. If absent, both components of the format specifier are inferred from; the matching format of the numeric variable(s) used by the expression; constraint if any, and defaults to ``%u`` if no numeric variable is used,; denoting that the value should be unsigned with no leading zeros. In case of; conflict between format specifiers of several numeric variables, the; conversion specifier becomes mandatory but the precision specifier remains; optional. * ``<constraint>`` is the constraint de
**Answer:**
to_eliminate: true
reasoning: The content only consists of programmatic interface description and no other text.

### Example 3
**Content:** I've been testing the latest updates on our scientific computing framework, and I'm impressed by the improvements in performance. The new parallelization strategy has reduced our simulation times by about 30% on our cluster. However, I noticed that memory usage spikes significantly when handling large datasets. Perhaps we could explore more efficient data structures or caching mechanisms to mitigate this issue? @TeamLead, would it be possible to allocate some resources to investigate this further?
**Answer:**
to_eliminate: false
reasoning: The content contains meaningful human-written sentences in natural language

### Example 4
**Content:** I've been experimenting with the GPU acceleration patches in this PR, and the results are promising. We're seeing a significant boost in matrix operations, which is crucial for our machine learning models. However, I encountered some inconsistencies when running on different GPU architectures. It might be beneficial to add more comprehensive tests to ensure compatibility across various hardware configurations. @DevTeam, could we discuss implementing a more robust testing framework for this?
**Answer:**
to_eliminate: false
reasoning: The content contains meaningful human-written sentences in natural language

### Example 5
**Content:** I've been exploring the energy efficiency improvements in our latest scientific computing framework, and I'm excited about the potential savings. By implementing parallelization strategies and optimizing data structures, we've reduced energy consumption by approximately 25% on our cluster. However, I think we could further enhance this by integrating tools like EnergyMeter to monitor and optimize energy usage in real-time. Additionally, exploring autotuning frameworks like ytopt could help balance performance and energy efficiency across different hardware configurations. @DevTeam, would it be feasible to allocate resources for integrating these tools and conducting a comprehensive energy efficiency analysis?
**Answer:**
to_eliminate: false
reasoning: The content contains meaningful human-written sentences in natural language

### Example 6
**Content:** is relatively simple;; they ultimately form a linked list of every clobber that dominates the; ``MemoryAccess`` that you're trying to optimize. In other words, the; ``definingAccess`` of a ``MemoryDef`` is always the nearest dominating; ``MemoryDef`` or ``MemoryPhi`` of said ``MemoryDef``. Use and Def optimization; ------------------------. ``MemoryUse``\ s keep a single operand, which is their defining or optimized; access.; Traditionally ``MemorySSA`` optimized ``MemoryUse``\ s at build-time, up to a; given threshold.; Specifically, the operand of every ``MemoryUse`` was optimized to point to the; actual clobber of said ``MemoryUse``. This can be seen in the above example; the; second ``MemoryUse`` in ``if.end`` has an operand of ``1``, which is a; ``MemoryDef`` from the entry block. This is done to make walking,; value numbering, etc, faster and easier.; As of `this revision <https://reviews.llvm.org/D121381>`_, the default was; changed to not optimize uses at build time, in order to provide the option to; reduce compile-time if the walking is not necessary in a pass. Most users call; the new API ``ensureOptimizedUses()`` to keep the previous behavior and do a; one-time optimization of ``MemoryUse``\ s, if this was not done before.; New pass users are recommended to call ``ensureOptimizedUses()``. Initially it was not possible to optimize ``MemoryDef``\ s in the same way, as we; restricted ``MemorySSA`` to one operand per access.; This was changed and ``MemoryDef``\ s now keep two operands.; The first one, the defining access, is; always the previous ``MemoryDef`` or ``MemoryPhi`` in the same basic block, or; the last one in a dominating predecessor if the current block doesn't have any; other accesses writing to memory. This is needed for walking Def chains.; The second operand is the optimized access, if there was a previous call on the; walker's ``getClobberingMemoryAccess(MA)``. This API will cache information; as part of ``MA``.; Optimizing all ``MemoryDef``
**Answer:**
to_eliminate: false
reasoning: The content contains meaningful human-written sentences in natural language discussing testing experiences
and performance improvements.

### Example 7
**Content:** representations, but these models are targeted towards token- and sentence-level training objectives and do not leverage information on inter-document relatedness, which limits their document-level representation power.; For applications on scientific documents, such as classification and recommendation, the embeddings power strong performance on end tasks.; We propose SPECTER, a new method to generate document-level embedding of scientific documents based on pretraining a Transformer language model on a powerful signal of document-level relatedness: the citation graph.; Unlike existing pretrained language models, SPECTER can be easily applied to downstream applications without task-specific fine-tuning.;
**Answer:**
to_eliminate: false
reasoning: The content discusses research on NLP applications in scientific document processing, including the introduction of new methods like SPECTER and SCIDOCS.

---

## Now analyze ONLY the following content:

**Content to evaluate:**  
{x['sentence']}
"""

to_prompt = lambda x: \
f"""
You are an expert in analyzing and categorizing text content. Your task is to evaluate whether the given **target content** should be filtered out, based on whether it consists of meaningful human-written prose or instead mainly contains programmatic or technical artifacts like logs or code.

## Instructions:
For each input, return:
1. `to_eliminate`: true or false — should this content be eliminated?
2. `reasoning`: Brief explanation of why the decision was made.


### Eliminate content that is not intended for human interpretation and consists primarily of:
- Code snippets or program structure  
  *(e.g., `if/else`, `for` loops, braces, language-specific syntax or keywords)*
- Program output, logs, or error traces  
  *(e.g., timestamps, error codes, stack traces, unit test results)*
- Configuration files, scripts, or build system output  
  *(e.g., YAML, JSON, Makefiles, shell scripts, compiler output)*
- Version control metadata or commit messages  
  *(e.g., git logs, diffs, merge info, file paths with change indicators)*
- API documentation or technical interface definitions  
  *(e.g., method signatures, parameter tables, annotations, formal docstrings)*

### Keep content that:
- Does **not** fall primarily into any of the elimination categories
- Is written for human readers — including **natural language, explanation, commentary, or analysis**
- Includes **scientific, academic, or technical discussions**, even if highly formal or specialized  
  *(e.g., discussions of model architecture, training benchmarks, research outcomes, biological findings, or engineering analysis)*
- May contain structured or technical vocabulary, **as long as it is not formatted primarily like code or machine output**
- Reflects **communication intended for developers or users**, such as thoughtful suggestions, analysis, or critiques

## Examples (for reference only – do not analyze):

### Example 1
**Content:** Build failed on ROOT-ubuntu2004/python3.; Running on root-ubuntu-2004-3.cern.ch:/home/sftnight/build/workspace/root-pullrequests-build; [See console output](https://lcgapp-services.cern.ch/root-jenkins/job/root-pullrequests-build/183837/console).; ### Failing tests:; - [projectroot.test.test_stressgraphics_interpreted](https://lcgapp-services.cern.ch/root-jenkins/job/root-pullrequests-build/183837/testReport/projectroot/test/test_stressgraphics_interpreted/)
**Answer:**
to_eliminate: true
reasoning: The content only consists of logs and no other text.

### Example 2
**Content:** recision><conversion specifier>`` where:. * ``#`` is an optional flag available for hex values (see; ``<conversion specifier>`` below) which requires the value matched to be; prefixed by ``0x``.; * ``.<precision>`` is an optional printf-style precision specifier in which; ``<precision>`` indicates the minimum number of digits that the value matched; must have, expecting leading zeros if needed. * ``<conversion specifier>`` is an optional scanf-style conversion specifier; to indicate what number format to match (e.g. hex number). Currently; accepted format specifiers are ``%u``, ``%d``, ``%x`` and ``%X``. If absent,; the format specifier defaults to ``%u``. For example:. .. code-block:: llvm. ; CHECK: mov r[[#REG:]], 0x[[#%.8X,ADDR:]]. would match ``mov r5, 0x0000FEFE`` and set ``REG`` to the value ``5`` and; ``ADDR`` to the value ``0xFEFE``. Note that due to the precision it would fail; to match ``mov r5, 0xFEFE``. As a result of the numeric variable definition being optional, it is possible; to only check that a numeric value is present in a given format. This can be; useful when the value itself is not useful, for instance:. .. code-block:: gas. ; CHECK-NOT: mov r0, r[[#]]. to check that a value is synthesized rather than moved around. The syntax of a numeric substitution is; ``[[#%<fmtspec>, <constraint> <expr>]]`` where:. * ``<fmtspec>`` is the same format specifier as for defining a variable but; in this context indicating how a numeric expression value should be matched; against. If absent, both components of the format specifier are inferred from; the matching format of the numeric variable(s) used by the expression; constraint if any, and defaults to ``%u`` if no numeric variable is used,; denoting that the value should be unsigned with no leading zeros. In case of; conflict between format specifiers of several numeric variables, the; conversion specifier becomes mandatory but the precision specifier remains; optional. * ``<constraint>`` is the constraint de
**Answer:**
to_eliminate: true
reasoning: The content only consists of programmatic interface description and no other text.

### Example 3
**Content:** I've been testing the latest updates on our scientific computing framework, and I'm impressed by the improvements in performance. The new parallelization strategy has reduced our simulation times by about 30% on our cluster. However, I noticed that memory usage spikes significantly when handling large datasets. Perhaps we could explore more efficient data structures or caching mechanisms to mitigate this issue? @TeamLead, would it be possible to allocate some resources to investigate this further?
**Answer:**
to_eliminate: false
reasoning: The content contains meaningful human-written sentences in natural language

### Example 4
**Content:** I've been experimenting with the GPU acceleration patches in this PR, and the results are promising. We're seeing a significant boost in matrix operations, which is crucial for our machine learning models. However, I encountered some inconsistencies when running on different GPU architectures. It might be beneficial to add more comprehensive tests to ensure compatibility across various hardware configurations. @DevTeam, could we discuss implementing a more robust testing framework for this?
**Answer:**
to_eliminate: false
reasoning: The content contains meaningful human-written sentences in natural language

### Example 5
**Content:** I've been exploring the energy efficiency improvements in our latest scientific computing framework, and I'm excited about the potential savings. By implementing parallelization strategies and optimizing data structures, we've reduced energy consumption by approximately 25% on our cluster. However, I think we could further enhance this by integrating tools like EnergyMeter to monitor and optimize energy usage in real-time. Additionally, exploring autotuning frameworks like ytopt could help balance performance and energy efficiency across different hardware configurations. @DevTeam, would it be feasible to allocate resources for integrating these tools and conducting a comprehensive energy efficiency analysis?
**Answer:**
to_eliminate: false
reasoning: The content contains meaningful human-written sentences in natural language

### Example 6
**Content:** is relatively simple;; they ultimately form a linked list of every clobber that dominates the; ``MemoryAccess`` that you're trying to optimize. In other words, the; ``definingAccess`` of a ``MemoryDef`` is always the nearest dominating; ``MemoryDef`` or ``MemoryPhi`` of said ``MemoryDef``. Use and Def optimization; ------------------------. ``MemoryUse``\ s keep a single operand, which is their defining or optimized; access.; Traditionally ``MemorySSA`` optimized ``MemoryUse``\ s at build-time, up to a; given threshold.; Specifically, the operand of every ``MemoryUse`` was optimized to point to the; actual clobber of said ``MemoryUse``. This can be seen in the above example; the; second ``MemoryUse`` in ``if.end`` has an operand of ``1``, which is a; ``MemoryDef`` from the entry block. This is done to make walking,; value numbering, etc, faster and easier.; As of `this revision <https://reviews.llvm.org/D121381>`_, the default was; changed to not optimize uses at build time, in order to provide the option to; reduce compile-time if the walking is not necessary in a pass. Most users call; the new API ``ensureOptimizedUses()`` to keep the previous behavior and do a; one-time optimization of ``MemoryUse``\ s, if this was not done before.; New pass users are recommended to call ``ensureOptimizedUses()``. Initially it was not possible to optimize ``MemoryDef``\ s in the same way, as we; restricted ``MemorySSA`` to one operand per access.; This was changed and ``MemoryDef``\ s now keep two operands.; The first one, the defining access, is; always the previous ``MemoryDef`` or ``MemoryPhi`` in the same basic block, or; the last one in a dominating predecessor if the current block doesn't have any; other accesses writing to memory. This is needed for walking Def chains.; The second operand is the optimized access, if there was a previous call on the; walker's ``getClobberingMemoryAccess(MA)``. This API will cache information; as part of ``MA``.; Optimizing all ``MemoryDef``
**Answer:**
to_eliminate: false
reasoning: The content contains meaningful human-written sentences in natural language discussing testing experiences
and performance improvements.

### Example 7
**Content:** representations, but these models are targeted towards token- and sentence-level training objectives and do not leverage information on inter-document relatedness, which limits their document-level representation power.; For applications on scientific documents, such as classification and recommendation, the embeddings power strong performance on end tasks.; We propose SPECTER, a new method to generate document-level embedding of scientific documents based on pretraining a Transformer language model on a powerful signal of document-level relatedness: the citation graph.; Unlike existing pretrained language models, SPECTER can be easily applied to downstream applications without task-specific fine-tuning.;
**Answer:**
to_eliminate: false
reasoning: The content discusses research on NLP applications in scientific document processing, including the introduction of new methods like SPECTER and SCIDOCS.

---

## Now analyze ONLY the following content:

**Content to evaluate:**  
{x['sentence']}
"""

def cleanup_and_exit(signal_num, frame):
    print("Caught interrupt, cleaning up...")
    sys.exit(0)  # Triggers the context manager's cleanup

# Register the signal handler
signal.signal(signal.SIGINT, cleanup_and_exit)

MIN_WORD_COUNT = 10

def filter_df(df):
    df = df[df["quality_attribute"].isin(["Testability", "Energy Efficiency"])]
    df["word_count"] = df["sentence"].apply(lambda x: len(re.sub(r"[\W_]+", " ", x).strip().split()))
    df = df[df["word_count"] >= MIN_WORD_COUNT]
    return df


def verify_file_batched_llm(file_path: Path, res_filepath: Path, host: str, batch_size=10):
    os.makedirs(f".cache/{FolderNames.FORMAT_VALIDATION_DIR}/", exist_ok=True)
    with shelve.open(f".cache/{FolderNames.FORMAT_VALIDATION_DIR}/{file_path.stem}") as db:
        if db.get("processed", False):
            logger.info(f"File {file_path.stem} already processed")
            return
        logger.info(f"Processing {file_path.stem}")

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            logger.info(e)
            return

        # df = df[df["true_positive"] == True]
        # df = filter_df(df)

        last_idx = db.get("idx", 0)
        df = df.iloc[last_idx:].copy()
        if last_idx > 0:
            logger.info(f"Continuing from {last_idx}")
            res_filepath = res_filepath.with_suffix(f".from_{last_idx}.csv")

        df['format_prompt'] = df.apply(lambda x: to_prompt(x), axis=1)
        res = []

        for i in tqdm(range(0, len(df), batch_size), total=math.ceil(len(df) / batch_size), desc=f"Verifying {file_path.stem} in batches of {batch_size}"):
            batch_df = df.iloc[i:i + batch_size]
            prompts = batch_df['format_prompt'].tolist()

            try:
                responses = request_ollama_chain(prompts, host)  # New batch query
                processed_responses = [(r.to_eliminate, r.reason) for r in responses]
                res.extend(processed_responses)
            except RetryError as error:
                logger.error(f"Retry error at batch starting index {last_idx + i}, {error}")
                responses = [(None, str(error))] * len(batch_df)
                res.extend(responses)
            except Exception as e:
                logger.error(e)
                errors_for_termination = ["HTTPConnectionPool",
                                          "No connection could be made because the target machine actively refused it"]
                if any(error in str(e) for error in errors_for_termination):
                    logger.error("HTTPConnectionPool error, exiting")
                    exit(1)
                responses = [(None, str(e))] * len(batch_df)
                res.extend(responses)

            df_to_save = df.iloc[:i + batch_size].copy()
            df_to_save['to_eliminate'], df_to_save['reason'] = zip(*res)
            df_to_save.to_csv(res_filepath, index=False)
            db["idx"] = last_idx + i + batch_size

        db['processed'] = True
        logger.info(f"Processed {file_path.stem}")


def validate_arch(host, only_files_containing_text: List[str] | None = None, reverse: bool = False):
    only_files_containing_text = only_files_containing_text or []
    keyword_folder = Path("metadata/keywords/")
    optimized_keyword_folder = keyword_folder / FolderNames.OPTIMIZED_KEYWORD_DIR
    os.makedirs(".logs", exist_ok=True)
    os.makedirs(keyword_folder / FolderNames.FORMAT_VALIDATION_DIR, exist_ok=True)
    logger.add(create_logger_path(FolderNames.FORMAT_VALIDATION_DIR), mode="w")

    try:
        for file_path in optimized_keyword_folder.glob("*.csv"):
            if any(cred.get_ref(".") in file_path.stem for cred in (credential_list)):
                keep_processing = len(only_files_containing_text) == 0 or any(text_to_test in file_path.stem for text_to_test in only_files_containing_text)
                if keep_processing == reverse:
                    continue

                res_filepath = keyword_folder / f"{FolderNames.FORMAT_VALIDATION_DIR}/{file_path.stem}.arch_verified.csv"
                verify_file_batched_llm(file_path, res_filepath, host,
                                        10)  # res_filepath = file_path.with_stem("test123")
    except Exception as e:
        logger.error(e)
        raise e


LOCAL_LLM_HOST = "http://localhost:11434"

if __name__ == "__main__":
    validate_arch(LOCAL_LLM_HOST, [
        "root-project"
    ], reverse=True)
