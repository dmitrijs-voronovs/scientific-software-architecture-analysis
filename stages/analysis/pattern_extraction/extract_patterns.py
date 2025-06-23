import dataclasses
import json
import os
import re
import shelve
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Dict

import requests
from loguru import logger
from pandas import DataFrame
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm

from processing_pipeline.keyword_matching.extract_quality_attribs_from_docs import KeywordParser
from model.Credentials import Credentials
from utils.utils import create_logger_path


@dataclass
class PatternSignature:
    name: str
    roles: List[str]  # e.g., ["Subject", "Observer"] for Observer pattern
    relationships: List[str]  # e.g., ["Subject notifies Observer"]
    common_names: List[str]  # e.g., ["EventManager", "Subscriber"]
    common_methods: List[str]  # e.g., ["attach", "detach", "notify"]
    common_terms: List[str]  # Domain-specific vocabulary


patterns: Dict[str, PatternSignature] = {"observer": PatternSignature(name="Observer", roles=["Subject", "Observer"],
                                                                      relationships=[
                                                                          "Subject maintains list of observers",
                                                                          "Subject notifies observers",
                                                                          "Observers update based on subject state"],
                                                                      common_names=["EventManager", "Publisher",
                                                                                    "Subject", "Observable",
                                                                                    "Subscriber", "Observer",
                                                                                    "Listener"],
                                                                      common_methods=["attach", "detach", "notify",
                                                                                      "update", "on_change",
                                                                                      "subscribe", "unsubscribe"],
                                                                      common_terms=["event", "notification",
                                                                                    "subscription", "broadcast",
                                                                                    "listener", "handler"]),
                                         "factory": PatternSignature(name="Factory", roles=["Creator", "Product"],
                                                                     relationships=["Creator instantiates Products",
                                                                                    "Products share interface"],
                                                                     common_names=["Factory", "Creator", "Builder",
                                                                                   "Generator", "Product", "Concrete"],
                                                                     common_methods=["create", "build", "make",
                                                                                     "generate", "get_instance"],
                                                                     common_terms=["factory", "creation",
                                                                                   "instantiation", "product"]),
                                         "singleton": PatternSignature(name="Singleton", roles=["Singleton"],
                                                                       relationships=[
                                                                           "Ensures a class has only one instance",
                                                                           "Provides global access to the instance"],
                                                                       common_names=["Singleton", "Instance",
                                                                                     "GlobalInstance"],
                                                                       common_methods=["get_instance", "instance",
                                                                                       "create_instance"],
                                                                       common_terms=["single instance", "global state",
                                                                                     "shared resource"]),
                                         "strategy": PatternSignature(name="Strategy", roles=["Context", "Strategy"],
                                                                      relationships=[
                                                                          "Context uses a Strategy interface",
                                                                          "Concrete Strategies implement Strategy interface"],
                                                                      common_names=["Strategy", "Policy", "Algorithm",
                                                                                    "Handler"],
                                                                      common_methods=["execute", "perform", "run",
                                                                                      "apply"],
                                                                      common_terms=["strategy", "policy", "behavior",
                                                                                    "algorithm"]),
                                         "decorator": PatternSignature(name="Decorator",
                                                                       roles=["Component", "Decorator"],
                                                                       relationships=["Decorators wrap Components",
                                                                                      "Decorators extend functionality of Components"],
                                                                       common_names=["Decorator", "Wrapper",
                                                                                     "Component", "Enhancer"],
                                                                       common_methods=["wrap", "decorate",
                                                                                       "add_behavior"],
                                                                       common_terms=["decoration", "wrapping",
                                                                                     "enhancement", "extension"]),
                                         "adapter": PatternSignature(name="Adapter",
                                                                     roles=["Client", "Adapter", "Adaptee"],
                                                                     relationships=[
                                                                         "Adapter translates requests from Client to Adaptee",
                                                                         "Adapter implements Client-compatible interface"],
                                                                     common_names=["Adapter", "Wrapper", "Translator"],
                                                                     common_methods=["adapt", "translate", "convert",
                                                                                     "get_compatible"],
                                                                     common_terms=["adaptation", "translation",
                                                                                   "compatibility", "conversion"]),
                                         "composite": PatternSignature(name="Composite",
                                                                       roles=["Component", "Leaf", "Composite"],
                                                                       relationships=["Composite contains Components",
                                                                                      "Leaf is a Component without children"],
                                                                       common_names=["Composite", "Leaf", "Component",
                                                                                     "Node"],
                                                                       common_methods=["add", "remove", "get_child",
                                                                                       "operation"],
                                                                       common_terms=["hierarchy", "tree structure",
                                                                                     "composition", "aggregate"]),
                                         "command": PatternSignature(name="Command",
                                                                     roles=["Invoker", "Command", "Receiver"],
                                                                     relationships=["Invoker calls Command",
                                                                                    "Command executes actions on Receiver"],
                                                                     common_names=["Command", "Action", "Task",
                                                                                   "Invoker", "Executor", "Receiver"],
                                                                     common_methods=["execute", "run", "perform",
                                                                                     "undo", "redo"],
                                                                     common_terms=["command", "action", "execution",
                                                                                   "task"]),
                                         "builder": PatternSignature(name="Builder",
                                                                     roles=["Builder", "Director", "Product"],
                                                                     relationships=[
                                                                         "Director constructs a Product using Builder",
                                                                         "Builder defines steps to create a Product"],
                                                                     common_names=["Builder", "Director", "Assembler",
                                                                                   "Constructor"],
                                                                     common_methods=["build", "construct", "assemble",
                                                                                     "create"],
                                                                     common_terms=["construction", "step-by-step",
                                                                                   "assembly"]),
                                         "proxy": PatternSignature(name="Proxy",
                                                                   roles=["Client", "Proxy", "RealSubject"],
                                                                   relationships=[
                                                                       "Proxy controls access to RealSubject",
                                                                       "Proxy provides the same interface as RealSubject"],
                                                                   common_names=["Proxy", "Placeholder", "Surrogate",
                                                                                 "Wrapper"],
                                                                   common_methods=["request", "access", "handle"],
                                                                   common_terms=["proxy", "surrogate", "control access",
                                                                                 "representation"]),
                                         "state": PatternSignature(name="State", roles=["Context", "State"],
                                                                   relationships=[
                                                                       "Context maintains a reference to a State object",
                                                                       "State defines behavior associated with Context's state"],
                                                                   common_names=["State", "Mode", "Context"],
                                                                   common_methods=["transition_to", "handle",
                                                                                   "change_state", "execute"],
                                                                   common_terms=["state", "mode", "context",
                                                                                 "behavior"]),
                                         "template_method": PatternSignature(name="Template Method",
                                                                             roles=["AbstractClass", "ConcreteClass"],
                                                                             relationships=[
                                                                                 "AbstractClass defines template of an algorithm",
                                                                                 "ConcreteClass overrides steps of the algorithm"],
                                                                             common_names=["Template", "AbstractClass",
                                                                                           "BaseClass"],
                                                                             common_methods=["template_method", "step1",
                                                                                             "step2", "customize"],
                                                                             common_terms=["template", "skeleton",
                                                                                           "algorithm steps"]),
                                         # Creational Patterns
                                         "abstract_factory": PatternSignature(name="Abstract Factory",
                                                                              roles=["Factory", "Product"],
                                                                              relationships=[
                                                                                  "Abstract Factory defines an interface for creating families of products",
                                                                                  "Concrete Factories implement creation for specific product families"],
                                                                              common_names=["AbstractFactory",
                                                                                            "Factory", "ProductFamily",
                                                                                            "FactoryMethod"],
                                                                              common_methods=["create_product_a",
                                                                                              "create_product_b",
                                                                                              "get_factory"],
                                                                              common_terms=["factory", "product family",
                                                                                            "abstract factory"]),
                                         "prototype": PatternSignature(name="Prototype", roles=["Prototype"],
                                                                       relationships=[
                                                                           "Prototype creates new objects by copying an existing object"],
                                                                       common_names=["Prototype", "Cloneable", "Copy"],
                                                                       common_methods=["clone", "copy", "duplicate"],
                                                                       common_terms=["prototyping", "copying",
                                                                                     "cloning", "duplication"]),

                                         # Structural Patterns
                                         "bridge": PatternSignature(name="Bridge",
                                                                    roles=["Abstraction", "Implementation"],
                                                                    relationships=[
                                                                        "Abstraction uses an Implementation interface",
                                                                        "Implementation provides the concrete behavior"],
                                                                    common_names=["Bridge", "Abstraction",
                                                                                  "Implementation", "Refinement"],
                                                                    common_methods=["operation", "implement", "extend",
                                                                                    "refine"],
                                                                    common_terms=["bridge", "abstraction", "decoupling",
                                                                                  "implementation"]),
                                         "facade": PatternSignature(name="Facade", roles=["Facade", "Subsystem"],
                                                                    relationships=[
                                                                        "Facade provides a simplified interface to a subsystem",
                                                                        "Subsystem performs the actual operations"],
                                                                    common_names=["Facade", "Wrapper", "Interface",
                                                                                  "API"],
                                                                    common_methods=["initialize", "run", "simplify",
                                                                                    "execute"],
                                                                    common_terms=["facade", "interface",
                                                                                  "simplification", "wrapper"]),
                                         "flyweight": PatternSignature(name="Flyweight", roles=["Flyweight", "Context"],
                                                                       relationships=[
                                                                           "Flyweight shares common data across multiple contexts",
                                                                           "Context uses Flyweight to reduce memory usage"],
                                                                       common_names=["Flyweight", "SharedState",
                                                                                     "IntrinsicState",
                                                                                     "ExtrinsicState"],
                                                                       common_methods=["get_instance", "render",
                                                                                       "reuse"],
                                                                       common_terms=["flyweight", "shared state",
                                                                                     "memory optimization"]),

                                         # Behavioral Patterns
                                         "chain_of_responsibility": PatternSignature(name="Chain of Responsibility",
                                                                                     roles=["Handler"], relationships=[
                                                 "Handlers are linked to form a chain",
                                                 "Request passes along the chain until a handler processes it"],
                                                                                     common_names=["Handler",
                                                                                                   "RequestProcessor",
                                                                                                   "Middleware"],
                                                                                     common_methods=["handle",
                                                                                                     "set_next",
                                                                                                     "process",
                                                                                                     "pass_to_next"],
                                                                                     common_terms=["chain",
                                                                                                   "responsibility",
                                                                                                   "middleware",
                                                                                                   "pipeline"]),
                                         "iterator": PatternSignature(name="Iterator", roles=["Iterator", "Collection"],
                                                                      relationships=[
                                                                          "Iterator provides sequential access to elements in a collection",
                                                                          "Collection creates Iterator instances"],
                                                                      common_names=["Iterator", "Cursor", "Enumerator"],
                                                                      common_methods=["next", "has_next", "reset",
                                                                                      "current"],
                                                                      common_terms=["iteration", "traversal",
                                                                                    "sequential access"]),
                                         "mediator": PatternSignature(name="Mediator", roles=["Mediator", "Colleague"],
                                                                      relationships=[
                                                                          "Mediator centralizes communication between Colleagues",
                                                                          "Colleagues communicate through the Mediator"],
                                                                      common_names=["Mediator", "Controller",
                                                                                    "Coordinator", "CentralHub"],
                                                                      common_methods=["mediate", "notify", "coordinate",
                                                                                      "send_message"],
                                                                      common_terms=["mediator", "coordination",
                                                                                    "central communication"]),
                                         "memento": PatternSignature(name="Memento",
                                                                     roles=["Originator", "Memento", "Caretaker"],
                                                                     relationships=[
                                                                         "Originator creates Memento to capture state",
                                                                         "Caretaker stores and restores Memento"],
                                                                     common_names=["Memento", "Snapshot", "Caretaker",
                                                                                   "StateSaver"],
                                                                     common_methods=["save_state", "restore_state",
                                                                                     "create_memento", "get_memento"],
                                                                     common_terms=["memento", "state", "snapshot",
                                                                                   "undo"]),
                                         "visitor": PatternSignature(name="Visitor", roles=["Visitor", "Element"],
                                                                     relationships=[
                                                                         "Visitor adds operations to Element without modifying it",
                                                                         "Element accepts Visitors"],
                                                                     common_names=["Visitor", "Element", "Operation",
                                                                                   "AcceptVisitor"],
                                                                     common_methods=["visit", "accept", "apply"],
                                                                     common_terms=["visitor", "operation", "element",
                                                                                   "acceptance"])

                                         }

LOCAL_LLM_HOST = "http://localhost:11434"


@retry(stop=stop_after_attempt(6), wait=wait_fixed(3), after=lambda retry_state: logger.warning(retry_state),
       reraise=True, )
def request_gemma(prompt):
    url = "%s/api/generate" % LOCAL_LLM_HOST

    payload = json.dumps({"model": "gemma", "prompt": prompt, "stream": False})
    response = requests.request("POST", url, headers={'Content-Type': 'application/json'}, data=payload).json()

    try:
        text_resp = re.sub(r'```json|```', "", response['response'])
        json_resp = json.loads(text_resp)
        return json_resp
    except Exception as e:
        raise Exception(f"Error in response: {response['response']}", e)


to_file_prompt = lambda filename, source_code: f"""
You are a software design pattern expert and a skilled code analyst. Analyze the file provided below and perform the following tasks:

1. Design Pattern Extraction: Identify and evaluate occurrences of any software design patterns from the following list:
```
{[dataclasses.asdict(val) for val in patterns.values()]}
```
 
   - For each identified pattern:
     - State whether it is a full match or partial match.
     - Highlight which parts of the pattern are implemented (e.g., specific roles, methods, or relationships).
     - Assign a confidence score between 0 and 1 based on how closely the code matches the pattern.
     - Provide specific evidence supporting the match, such as method names, relationships, or terminology used in the source code.

2. File Description: Provide a high-level description of what this file does. 
   - Include details of the file's primary purpose or functionality.
   - Summarize all constants, types, classes, and functions present in the file at a high level, explaining their roles and relationships briefly.

Input:
- Filename: {filename}
- Source Code:
```
{source_code}
```

Output:
- You must output only the JSON object as described below.
- Do not include any additional explanation, notes, or extra information beyond the JSON object.
- Do not add **Explanation** section at the end of the JSON object.
- Remove any additioanl sections from the end of the response
- Adhere strictly to the following structure for the output:

```json
{{
  "patterns": [
    {{
      "name": "<Pattern Name>",
      "match_type": "<full|partial>",
      "implemented_parts": ["<list of roles, methods, or relationships implemented>"],
      "confidence": <confidence_score>,
      "evidence": ["<list of specific evidence from the code>"]
    }}
  ],
  "description": {{
    "purpose": "<High-level description of the file's functionality>",
    "summary": {{
      "constants": ["<High-level description of constants>"],
      "types": ["<High-level description of types>"],
      "classes": ["<High-level description of classes>"],
      "functions": ["<High-level description of functions>"]
    }}
  }}
}}
```

Example Output:
```json
{{
  "patterns": [
    {{
      "name": "Observer",
      "match_type": "partial",
      "implemented_parts": ["Subject", "Observer", "notify method"],
      "confidence": 0.8,
      "evidence": ["class Subject", "notify observers", "on_change handler"]
    }},
    {{
      "name": "Singleton",
      "match_type": "full",
      "implemented_parts": ["Singleton instance", "get_instance method"],
      "confidence": 1.0,
      "evidence": ["private constructor", "static get_instance"]
    }}
  ],
  "description": {{
    "purpose": "This file implements a notification system for managing event listeners.",
    "summary": {{
      "constants": ["Defines event-related constants"],
      "types": ["Custom types for event handling"],
      "classes": ["Subject class manages observers", "Observer class represents listeners"],
      "functions": ["notify updates observers", "attach adds listeners"]
    }}
  }}
}}
```
"""

to_directory_prompt = lambda filename, context_data: f"""
Here’s a second prompt template designed for aggregating hierarchical results from multiple files or directories, building upon the output of the first prompt:

---

Prompt Template:

```
You are a software design pattern expert and a skilled code analyst. Analyze the provided directory structure to produce a combined report based on the results of individual files or subdirectories. Use the following context:

- Patterns: 
```
{patterns.values()}
```
- Context Data: This includes results of the analysis for individual files or subdirectories in the same format as your output for the first prompt.

Tasks:

1. Pattern Aggregation: Combine and summarize design pattern matches from the files and subdirectories:
   - Group results hierarchically, clearly showing which patterns are detected at the directory level and where they were found in the child files or directories.
   - For each pattern:
     - State whether it is a full match or partial match across the directory.
     - Highlight the aggregated implemented parts across all files/subdirectories.
     - Assign a combined confidence score as an average of the confidence scores across matches for this pattern.
     - Provide specific evidence summarized across all files/subdirectories, such as method names, relationships, or terminology.

2. Directory Description: Provide a high-level description of what the directory as a whole represents:
   - Summarize the overall purpose of the directory.
   - Provide aggregated high-level descriptions of constants, types, classes, and functions defined across all files/subdirectories.

Input:
- Directory Name: {filename}
- Context Data:
```
{context_data}
```

Output:
- You must output only the JSON object as described below.
- Do not include any additional explanation, notes, or extra information beyond the JSON object.
- Do not add **Explanation** section at the end of the JSON object.
- Remove any additioanl sections from the end of the response
- Adhere strictly to the following structure for the output:

```json
{{
  "patterns": [
    {{
      "name": "<Pattern Name>",
      "match_type": "<full|partial>",
      "implemented_parts": ["<list of roles, methods, or relationships aggregated from files or subdirectories>"],
      "confidence": <average_confidence_score>,
      "evidence": ["<list of summarized evidence across files and subdirectories>"]
    }}
  ],
  "description": {{
    "purpose": "<High-level description of the directory's purpose and functionality>",
    "summary": {{
      "constants": ["<High-level description of aggregated constants>"],
      "types": ["<High-level description of aggregated types>"],
      "classes": ["<High-level description of aggregated classes>"],
      "functions": ["<High-level description of aggregated functions>"]
    }}
  }}
}}
```

Example Output:
```json
{{
  "patterns": [
    {{
      "name": "Observer",
      "match_type": "partial",
      "implemented_parts": ["Subject", "Observer", "notify method"],
      "confidence": 0.75,
      "evidence": ["class Subject in file1.py", "notify method in file2.py", "on_change handler in file3.py"]
    }},
    {{
      "name": "Singleton",
      "match_type": "full",
      "aggregated_implemented_parts": ["Singleton instance", "get_instance method"],
      "confidence": 0.9,
      "evidence": ["private constructor in file4.py", "static get_instance in file5.py"]
    }}
  ],
  "description": {{
    "purpose": "This directory contains modules for managing an event notification system and a global configuration manager.",
    "summary": {{
      "constants": ["Defines shared constants for events and configuration"],
      "types": ["Custom types for events and global settings"],
      "classes": ["EventManager class for managing observers", "ConfigSingleton for global configuration"],
      "functions": ["notify updates observers", "get_instance returns singleton instance"]
    }}
  }}
}}
```
"""


class FileType(Enum):
    DIR = "dir"
    FILE = "file"


def pattern_extractor_iterator(creds: Credentials):
    repo_url = KeywordParser.get_github_repo_url(creds)
    source_code_dir = Path("../../../.tmp") / "source" / creds.get_ref()
    cache_dir = Path(".cache/patterns")
    os.makedirs(cache_dir, exist_ok=True)
    with shelve.open(str(cache_dir / creds.dotted_ref)) as db:
        last_processed_root = db.get('last')
        for root, dirs, files in os.walk(source_code_dir, topdown=False):
            if last_processed_root and root != last_processed_root:
                continue

            file_patterns = []
            try:
                for file in tqdm(files, desc=f"Extracting patterns from {root}"):
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.normpath(os.path.relpath(abs_path, source_code_dir)).replace("\\", "/")
                    with open(abs_path, "r", encoding="utf-8") as f:
                        try:
                            source_code = f.read()
                            prompt = to_file_prompt(abs_path, source_code)
                            result = request_gemma(prompt)
                            file_patterns.append(result)
                            # rel_path = os.path.normpath(os.path.relpath(abs_path, source_code_path)).replace("\\", "/")
                            yield dict(filename=rel_path, type=FileType.FILE, **creds,
                                       url=KeywordParser.generate_link(repo_url, rel_path), patterns=result["patterns"],
                                       **result["description"])
                        except Exception as e:
                            logger.error(f"Retry error, current_element={abs_path}, {prompt=}, {e=}")
            except Exception as e:
                logger.error(e)

            aggregated_prompt = to_directory_prompt(root, file_patterns)
            try:
                folder_result = request_gemma(aggregated_prompt)
                rel_folder_path = os.path.normpath(os.path.relpath(root, source_code_dir)).replace("\\", "/")
                yield dict(filename=rel_folder_path, type=FileType.DIR, **creds,
                           url=KeywordParser.generate_link(repo_url, rel_folder_path), patterns=folder_result["patterns"],
                           **folder_result["description"])
                db['last'] = root
            except Exception as e:
                logger.error(f"Retry error, current_element={root}, {prompt=}, {e=}")


pattern_extraction_dir = "pattern_extraction"


def extract_patterns(cred: Credentials, batch_size=20):
    save_destination = Path("metadata/patterns") / cred.get_ref(".")
    os.makedirs(save_destination, exist_ok=True)
    batch = []
    for i, pattern in enumerate(pattern_extractor_iterator(cred)):
        batch.append(pattern)
        if (i + 1) % batch_size == 0:
            df = DataFrame(batch)
            df.to_csv(save_destination / f"patterns.{i:04d}.csv", index=False)
            batch.clear()
    if len(batch) > 0:
        df = DataFrame(batch)
        df.to_csv(save_destination / f"patterns.-1.csv", index=False)


def main():
    keyword_folder = Path("metadata/keywords/")
    os.makedirs(Path("metadata/patterns"), exist_ok=True)
    os.makedirs("../../../.logs", exist_ok=True)

    os.makedirs(keyword_folder / pattern_extraction_dir, exist_ok=True)
    logger.add(create_logger_path(pattern_extraction_dir), mode="w")

    creds = [Credentials(
        {'author': 'scverse', 'repo': 'scanpy', 'version': '1.10.2', 'wiki': 'https://scanpy.readthedocs.io'}),
        Credentials({'author': 'allenai', 'repo': 'scispacy', 'version': 'v0.5.5',
                     'wiki': 'https://allenai.github.io/scispacy/'}),
        Credentials({'author': 'qutip', 'repo': 'qutip', 'version': 'v5.0.4', 'wiki': 'https://qutip.org'}),
        Credentials({'author': 'hail-is', 'repo': 'hail', 'version': '0.2.133', 'wiki': 'https://hail.is'}), ]

    try:
        for cred in creds:
            extract_patterns(cred)
    except Exception as e:
        logger.error(e)


if __name__ == "__main__":
    main()
