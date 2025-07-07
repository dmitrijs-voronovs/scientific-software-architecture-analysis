import io
import json
import re
from typing import TypedDict, Generator

import yaml

from servicess.pydantic_class_generator import tactic_description_iterator, TacticListDTO

generate_class_definition = lambda tactics: f"""
class TacticSimplifiedModel(BaseModel):
    tactic: Literal[{", ".join(tactics)}]
    response: str
"""

def generate_pydantic_classes(tactics: TacticListDTO):
    result = io.StringIO()
    result.write(
"""from pydantic import BaseModel
from typing import Literal
""")
    result.write(generate_class_definition([f'"{desc["tactic"]}"' for desc in tactic_description_iterator(tactics)]))
    return result.getvalue()

def main():
    with open("../processing_pipeline/s3_tactic_extraction/tactics/tactic_list.yaml", "r") as f:
        tactics: TacticListDTO = yaml.safe_load(f)
    result = generate_pydantic_classes(tactics)
    print(result)
    with open("../processing_pipeline/s3_tactic_extraction/tactics/tactic_list_simplified.py", "w") as f:
        f.write(result)

if __name__ == "__main__":
    main()