from pathlib import Path
from typing import Generator

import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Tree, Node

ext_to_lang = {"py": "python",  # "java": "java"
               }
Languages = {"python": Language(tspython.language()),  # "java": Language(tsjava.language())
             }


def extract_text_from_all_nodes(match):
    res = {}
    for [key, value] in match.items():
        res[key] = value[0].text.decode("utf-8")
    return res


def render_if_exists(object, field, prefix="", postfix="", orString="", render_field=lambda x: x):
    return prefix + render_field(object[field]) + postfix if field in object else orString


Queries = {"python": {"class_field": {"handler": (lambda match: {
    **(m := extract_text_from_all_nodes(match)),
    "embedding": f"Class field: {render_if_exists(m, "class.instance_field", "[instance] ", render_field=lambda x: "")}{m["class.name"]}.{m["class.field"].replace('self.', '')}"
}), "query": """
            (class_definition
                name: (identifier) @class.name
                body: (block
                    [ 
                        (expression_statement 
                            [
                                (assignment 
                                    left: (identifier) @field.name
                                    right: (_)? @field.default) @class.field
                                (assignment
                                    left: (_
                                        name: (identifier) @field.name
                                        type: (_) @field.type)
                                    right: (_)? @field.default) @class.field
                                (_
                                    name: (identifier) @field.name
                                    type: (_) @field.type) @class.field
                            ]
                        )
                        (function_definition
                            (identifier) @method.name
                            (#match @method.name "^__init__$")
                            (parameters) @method.parameters
                            (block
                                (expression_statement) @class.field @class.instance_field
                                (#match @class.instance_field "^self\\.")
                            )
                        )
                    ]
                )
            )
        """}, "class_method": {"handler": (lambda match: {
    **(m := extract_text_from_all_nodes(match)),
    "embedding": f"Class method: {render_if_exists(m, "method.decorator", "[", "] ")}{m["class.name"]}.{m["method.name"]}{m["method.parameters"]}{render_if_exists(m, "method.type", " -> ")}"
}), "query": """
            (class_definition
                name: (identifier) @class.name
                body: (block 
                    [
                        (function_definition
                            name: (identifier) @method.name
                            parameters: (parameters) @method.parameters
                            (type)? @method.type
                        )
                        (decorated_definition
                            (decorator (_) @method.decorator) 
                            (function_definition
                                name: (identifier) @method.name
                                parameters: (parameters) @method.parameters
                                (type)? @method.type
                            )
                        )
                    ]                        
                )
            ) ;; @class_method
        """},
                      "class": {"handler": (lambda match: {
                          **(m := extract_text_from_all_nodes(match)),
                          "embedding": f"Class: {m["class.name"]}{render_if_exists(m, "class.base")}"
                      }), "query": """
            (class_definition
                name: (identifier) @class.name
                superclasses: (argument_list)? @class.base
            ) ;; @class
        """},
                      "function": {"handler": (lambda match: {
                          **(m := extract_text_from_all_nodes(match)),
                          "embedding": f"Function: {m["function.name"]}{m["function.parameters"]}{render_if_exists(m, "function.type", " -> ")}"
                      }), "query": """
            (module
                (function_definition
                    name: (identifier) @function.name
                    parameters: (parameters) @function.parameters
                    (type)? @function.type
                    (comment)? @function.docstring
                ) ;; @function
            )
        """}, "module_type": {
        "handler": (lambda match: {
            **(m := extract_text_from_all_nodes(match)),
            "embedding": f"Type: {m['type.name']} = {m['type.value']}"
        }),
        "query": """
            (module
                (type_alias_statement
                    (type (identifier) @type.name)
                    (type (generic_type) @type.value)
                ) @type
            )
        """
    }, "constant": {
        "handler": (
            lambda match: {**(m := extract_text_from_all_nodes(match)), "embedding": f"Constant: {m["constant"]}"}),
        "query": """
            (module
                (expression_statement
                    (assignment
                        left: (identifier) @constant.name
                        right: (_) @constant.value
                    ) @constant
                )
            )
        """}, "import": {
        "handler": (lambda match: {
            **(m := extract_text_from_all_nodes(match)),
            "embedding": f"Import: {m["import.name"]}{render_if_exists(m, "import.from", " from ")}"
        }),
        "query": """
            (module
                (import_from_statement
                    module_name: (dotted_name) @import.from
                    (#match @import.from "^\\\\w+$")
                    name: (dotted_name)? @import.name
                    (aliased_import (dotted_name) @import.name)?
                ) @import
            )
            (module
                (import_statement
                    (dotted_name) @import.name
                ) @import
            )
            (module
                (import_statement
                    (aliased_import (dotted_name) @import.name)
                ) @import
            )
        """}, "local_import": {
        "handler": (lambda match: {
            **(m := extract_text_from_all_nodes(match)),
            "embedding": f"Import: {m["import.name"]} from {m["import.from"]}"
        }),
        "query": """
            (module
                (import_from_statement
                    (dotted_name
                        (identifier)
                        ("." (identifier))+
                    ) @import.from @import.path
                    name: (dotted_name)? @import.name
                    (aliased_import (dotted_name) @import.name)?
                ) @import
            )
        """}}}


def get_file_params(file_path: str) -> [str, str]:
    path = Path(file_path)
    return path.name, path.suffix


def read_file(file_path: str):
    name, ext = get_file_params(file_path)
    lang = ext_to_lang[ext[1:]]
    with open(file_path, "rb") as f:
        return f.read(), name, lang


def parse_code(code: bytes, lang: str):
    parser = Parser(Languages[lang])
    return parser.parse(code)


def traverse_tree(tree: Tree) -> Generator[Node, None, None]:
    cursor = tree.walk()

    visited_children = False
    level = 0
    while True:
        if not visited_children:
            yield cursor.node, level
            if cursor.goto_first_child():
                level += 1
            else:
                visited_children = True
        elif cursor.goto_next_sibling():
            visited_children = False
        elif cursor.goto_parent():
            level -= 1
        else:
            break


def extract_tree(tree: Tree, file):
    with open(file, "w") as f:
        for node, level in traverse_tree(tree):
            f.write(f"{'  ' * level}{node.type}: {node.text}\n")


def ast_iterator(lang, tree) -> Generator[dict, None, None]:
    for [query_name, config] in Queries[lang].items():
        handler, query_pattern = config["handler"], config["query"]
        # print(f"\nQuery: {query_name}, Pattern: {query_pattern}")
        print(f"\nQuery: {query_name}")
        query = Languages[lang].query(query_pattern)
        for [_, capture] in query.matches(tree.root_node):
            yield dict(element_type=query_name, **handler(capture))
