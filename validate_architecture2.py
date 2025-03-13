from validate_architecture import validate_arch

LOCAL_LLM_HOST = "http://localhost:11434"

if __name__ == "__main__":
    validate_arch(LOCAL_LLM_HOST, [
        "root-project.root.v6-32-06.WIKI.1",
        "root-project.root.v6-32-06.WIKI.2",
        "root-project.root.v6-32-06.WIKI.3",
        "root-project.root.v6-32-06.WIKI.4",
        "root-project.root.v6-32-06.WIKI.5",
        "root-project.root.v6-32-06.WIKI.6",
    ], True)
