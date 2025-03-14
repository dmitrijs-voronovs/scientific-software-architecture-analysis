import io
import json
import re
from typing import TypedDict, Generator

import yaml

class TacticDTO(dict):
    name: str
    description: str | None

class TacticCategoryDTO(dict):
    category_name: str
    tactics: list[TacticDTO]

class QualityAttributeDTO(dict):
    quality_attribute: str
    tactic_categories: list[TacticCategoryDTO]

class TacticListDTO(dict):
    tactics:  list[TacticDTO]

class TacticDefinition(TypedDict):
    class_name: str
    quality_attribute: str
    tactic_category: str
    tactic: str

class TacticDefinitionFull(TacticDefinition):
    description: str

end_result_sample = """
class Tactic(BaseModel):
    quality_attribute: Literal[quality_attribute]
    tactic_category: Literal[tactic_category]
    tactic: Literal[tactic]
    response: str

TacticCategory = Annotated[Tactic | Tactic | Tactic, Field(discriminator='tactic')]
QualityAttribute = Annotated[TacticCategory | TacticCategory | TacticCategory, Field(discriminator='tactic_category')]
class TacticsModel(BaseModel):
    architecture_tactic: Annotated[QualityAttribute | QualityAttribute | QualityAttribute, Field(discriminator='quality_attribute')]
"""

def to_camel_case(s: str, suffix: str = 'Model'):
    components = [f"{item[0].upper()}{item[1:].lower()}" for item in re.split(r'(?: \W?)+|-|/', s)]
    return ''.join(components) + suffix

def generate_tactic_class(x: TacticDefinition):
    return \
f"""
\nclass {x["class_name"]}(BaseModel):
    quality_attribute: Literal["{x["quality_attribute"]}"]
    tactic_category: Literal["{x["tactic_category"]}"]
    tactic: Literal["{x["tactic"]}"]
    response: str
"""

def generate_pydantic_classes(tactics: TacticListDTO):
    result = io.StringIO()
    result.write(
"""from pydantic import BaseModel, Field
from typing import Literal, Annotated
""")
    quality_attribute_classes_names = []
    for tactic in tactics["tactics"]:
        quality_attribute_class_name = to_camel_case(tactic["quality_attribute"])
        quality_attribute_classes_names.append(quality_attribute_class_name)
        tactic_category_names = []

        for tactic_category in tactic["tactic_categories"]:
            tactic_category_name =  to_camel_case(tactic_category["category_name"])
            tactic_category_names.append(tactic_category_name)
            tactic_names = []

            for tactic_definition in tactic_category["tactics"]:
                class_name = to_camel_case(tactic_definition["name"])
                tactic_names.append(class_name)
                result.write(generate_tactic_class(TacticDefinition(class_name=class_name, tactic_category=tactic_category["category_name"], quality_attribute=tactic["quality_attribute"], tactic=tactic_definition["name"])))

            result.write(f"\n\n{tactic_category_name} = Annotated[{" | ".join(tactic_names)}, Field(discriminator='tactic')]")
        result.write(f"\n\n{quality_attribute_class_name} = Annotated[{" | ".join(tactic_category_names)}, Field(discriminator='tactic_category')]")
    common_quality_attribute_class = \
f"""
\nclass TacticModel(BaseModel):
    architecture_tactic: Annotated[{" | ".join(quality_attribute_classes_names)}, Field(discriminator='quality_attribute')]
"""
    result.write(common_quality_attribute_class)
    return result.getvalue()

def tactic_description_iterator(tactics: TacticListDTO) -> Generator[TacticDefinitionFull, None, None]:
    for tactic in tactics["tactics"]:
        for tactic_category in tactic["tactic_categories"]:
            for tactic_definition in tactic_category["tactics"]:
                yield TacticDefinitionFull(class_name=to_camel_case(tactic_definition["name"]), tactic_category=tactic_category["category_name"], quality_attribute=tactic["quality_attribute"], tactic=tactic_definition["name"], description=tactic_definition["description"])


def get_descriptions(tactics: TacticListDTO):
    return {tactic["tactic"]: tactic["description"] for tactic in tactic_description_iterator(tactics)}

def main():
    with open("../cfg/tactic_list.yaml", "r") as f:
        tactics: TacticListDTO = yaml.safe_load(f)
    result = generate_pydantic_classes(tactics)
    print(result)
    with open("../cfg/tactic_list.py", "w") as f:
        f.write(result)

    descriptions = get_descriptions(tactics)
    with open("../cfg/tactic_description.py", "w") as f:
        f.write("tactic_descriptions = " + json.dumps(descriptions))

if __name__ == "__main__":
    main()