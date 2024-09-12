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


Queries = {"python": {"constants": {
    "handler": (lambda match: {**(m := extract_text_from_all_nodes(match)), "embedding": f"Constant: {m["constant"]}"}),
    "query": """
            (module
                (expression_statement
                    (assignment
                        left: (identifier) @constant.name
                        right: (_) @constant.value
                    ) @constant
                )
            )
        """}, "imports": {
    "handler": (lambda match: {**(m := extract_text_from_all_nodes(match)), "embedding": f"Import: {m["import"]}"}),
    "query": """
            (module
                (import_from_statement
                    module_name: (dotted_name)? @import.from
                    name: (dotted_name) @import.name) @import
            ;;    (import_statement
            ;;        name: (dotted_name) @import.name) @import
            )
        """}, "functions": {"handler": (lambda match: match), "query": """
            (module
                (function_definition
                    name: (identifier) @function.name
                    parameters: (parameters) @function.parameters) @function
            )
        """}, "classes": {"handler": (lambda match: match), "query": """
            (class_definition
                name: (identifier) @class.name
                superclasses: (argument_list)? @class.base) @class
        """}, "class_fields": {"handler": (lambda match: match), "query": """
            (class_definition
                name: (identifier) @class.name
                body: (block 
                    (expression_statement 
                        (assignment 
                            left: (identifier) @field.name
                            right: (_)? @field.default)))) @class_field
            (class_definition
                name: (identifier) @class.name
                body: (block 
                    (expression_statement
                        (assignment
                            left: (_
                                name: (identifier) @field.name
                                type: (_) @field.type)
                            right: (_)? @field.default)))) @class_field
            (class_definition
                name: (identifier) @class.name
                body: (block 
                    (expression_statement
                        (_
                            name: (identifier) @field.name
                            type: (_) @field.type)))) @class_field
        """}, "class_methods": {"handler": (lambda match: match), "query": """
            (class_definition
                name: (identifier) @class.name
                body: (block 
                    (function_definition
                        name: (identifier) @method.name
                        parameters: (parameters) @method.parameters))) @class_method
        """}}}


def get_ext(file_path: str):
    return Path(file_path).suffix


def read_file(file_path: str):
    ext = get_ext(file_path)
    lang = ext_to_lang[ext[1:]]
    with open(file_path, "rb") as f:
        return f.read(), lang


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


def main():
    code, lang = read_file("./tree_sitter_playground/test_file.py")
    tree = parse_code(code, lang)

    extract_tree(tree, "./tree_sitter_playground/test_file.tree")

    for [query_name, config] in Queries[lang].items():
        handler, query_pattern = config["handler"], config["query"]
        print(f"\nQuery: {query_name}, Pattern: {query_pattern}")
        query = Languages[lang].query(query_pattern)
        for [_, capture] in query.matches(tree.root_node):
            print(handler(capture))

        break


if __name__ == "__main__":
    main()
