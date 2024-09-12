from pathlib import Path
from typing import Generator

from tree_sitter import Language, Parser, Tree, Node
import tree_sitter_python as tspython
import tree_sitter_java as tsjava

ext_to_lang = {
    "py": "python",
    # "java": "java"
}

Languages = {
    "python": Language(tspython.language()),
    # "java": Language(tsjava.language())
}

Queries = {
    "python": {
        "constants": """
            (module
                (expression_statement
                    (assignment
                        left: (identifier) @constant.name
                        right: [
                            (integer) @constant.value
                            (float) @constant.value
                            (string) @constant.value
                            (true) @constant.value
                            (false) @constant.value
                            (call) @constant.value
                            (none) @constant.value
                        ]
                    ) @constant
                )
            )
            ;; Match constant names that are all uppercase and follow the pattern for constants
            ;; (#match? @constant.name "^[A-Z_][A-Z0-9_]*$")
        """,
        "imports": """
            (import_from_statement
                module_name: (dotted_name) @import.from
                name: (dotted_name) @import.name) @import
            (import_statement
                name: (dotted_name) @import.name) @import
        """,
        "functions": """
            (function_definition
                name: (identifier) @function.name
                parameters: (parameters) @function.parameters) @function
        """,
        "classes": """
            (class_definition
                name: (identifier) @class.name
                superclasses: (argument_list)? @class.base) @class
        """,
        "class_fields": """
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
        """,
        "class_methods": """
            (class_definition
                name: (identifier) @class.name
                body: (block 
                    (function_definition
                        name: (identifier) @method.name
                        parameters: (parameters) @method.parameters))) @class_method
        """
    }
}


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

    for [query_name, query_pattern] in Queries[lang].items():
        print(f"\nQuery: {query_name}, Pattern: {query_pattern}")
        query = Languages[lang].query(query_pattern)
        for capture in query.matches(tree.root_node):
            print(capture)

        break


if __name__ == "__main__":
    main()
