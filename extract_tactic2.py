from extract_tactic import extract_tactics

LOCAL_LLM_HOST = "http://localhost:11434"

if __name__ == "__main__":
    extract_tactics(LOCAL_LLM_HOST, ["root-project.root.v6-32-06."], True)
