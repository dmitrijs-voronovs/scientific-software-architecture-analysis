from services.ast_extractor import read_file, parse_code, extract_tree, ast_iterator


def main():
    file_path = "./tree_sitter_playground/test_file.py"
    # file_path = "ast-project-extractor.py"
    code, filename, lang = read_file(file_path)
    tree = parse_code(code, lang)

    extract_tree(tree, f"{filename}.tree")

    items = []
    for el in ast_iterator(lang, tree):
        items.append(dict(filename=filename, ext=lang, **el))
        print(el)

    # save into dataframes
    import pandas as pd
    df = pd.DataFrame(items)
    df.to_csv(f"{filename}.csv", index=False)


if __name__ == "__main__":
    main()
