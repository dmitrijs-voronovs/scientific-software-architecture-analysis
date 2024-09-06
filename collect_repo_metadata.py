from services.git import checkout_tag


def main():
    author = "scverse"
    repo_name = "scanpy"
    tag = "1.10.1"
    path = checkout_tag(author, repo_name, tag)


if __name__ == "__main__":
    main()
