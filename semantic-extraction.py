import spacy
from spacy.matcher import Matcher
import re
import json
from datetime import datetime


def load_tldrs(file_path):
    encodings = ['utf-8', 'utf-16', 'latin-1', 'ascii']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
            # Assuming TLDRs are separated by blank lines
            return content.split('\n')
        except UnicodeDecodeError:
            continue
    raise ValueError(
        f"Unable to decode the file with any of the following encodings: {encodings}")


def is_potential_project(doc):
    # Check for verbs related to development or research
    development_verbs = ["develop", "create", "build",
                         "implement", "design", "research", "study", "investigate"]
    if any(token.lemma_.lower() in development_verbs for token in doc):
        return True

    # Check for noun phrases related to technology or research
    tech_research_patterns = ["technology", "platform", "system",
                              "framework", "tool", "algorithm", "method", "approach"]
    if any(token.text.lower() in tech_research_patterns for token in doc.noun_chunks):
        return True

    # Check for specific entity types
    relevant_entities = ["ORG", "PRODUCT", "GPE", "WORK_OF_ART"]
    if any(ent.label_ in relevant_entities for ent in doc.ents):
        return True

    return False


def extract_project_info(doc):
    project_name = ''
    objective = ''

    # Extract project name
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART"] and not project_name:
            project_name = ent.text
            break

    if not project_name:
        for chunk in doc.noun_chunks:
            if any(token.pos_ == "PROPN" for token in chunk):
                project_name = chunk.text
                break

    # Extract objective
    for sent in doc.sents:
        if any(token.dep_ == 'ROOT' and token.pos_ == 'VERB' for token in sent):
            objective = sent.text
            break

    return project_name, objective


def process_tldrs(tldrs, nlp):
    projects = []

    for tldr in tldrs:
        doc = nlp(tldr)
        if is_potential_project(doc):
            project_name, objective = extract_project_info(doc)
            if project_name and objective:
                projects.append({
                    'name': project_name,
                    'objective': objective,
                    'tldr': tldr
                })

    return projects


def save_results_to_file(projects, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Found {len(projects)} potential projects:\n\n")
        for project in projects:
            f.write(f"Project Name: {project['name']}\n")
            f.write(f"Objective: {project['objective']}\n")
            f.write(f"TLDR: {project['tldr']}\n")
            f.write("-" * 50 + "\n\n")


def main():
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Load TLDRs from file
    tldrs = load_tldrs("./tldrs.txt")

    # Process TLDRs and extract potential projects
    projects = process_tldrs(tldrs, nlp)

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"potential_projects_{timestamp}.txt"

    # Save results to file
    save_results_to_file(projects, output_file)

    print(
        f"Found {len(projects)} potential projects. Results saved to {output_file}")


if __name__ == "__main__":
    main()
