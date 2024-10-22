from bs4 import BeautifulSoup

import os
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob

# Download required NLTK data
nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun


def lemmatize_text(text):
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    return ' '.join([lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags])


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return list(synonyms)


base_attributes = {
    'performance': ['fast', 'quick', 'speedy', 'efficient', 'responsive'],
    'security': ['secure', 'safe', 'protected', 'encrypted'],
    'maintainability': ['maintainable', 'clean', 'organized', 'structured'],
    'scalability': ['scalable', 'flexible', 'adaptable', 'extensible'],
    'reliability': ['reliable', 'stable', 'robust', 'dependable'],
    'usability': ['user-friendly', 'intuitive', 'easy-to-use', 'accessible'],
    'energy': []
}


def generate_quality_attributes():
    expanded_attributes = {}
    for attr, words in base_attributes.items():
        expanded_attributes[attr] = set(words)
        for word in words:
            expanded_attributes[attr].update(get_synonyms(word))

    return expanded_attributes


def generate_patterns(attributes):
    patterns = []
    for attr, words in attributes.items():
        for word in words:
            lemma = lemmatizer.lemmatize(word)
            patterns.extend([
                rf'\b{lemma}\b',
                rf'need(s)? (to be|for) \w+ {lemma}',
                rf'should be \w+ {lemma}',
                rf'must be \w+ {lemma}',
                rf'improve \w+ {lemma}',
                rf'enhance \w+ {lemma}',
                rf'optimize for \w+ {lemma}',
                rf'focus on \w+ {lemma}',
                rf'prioritize \w+ {lemma}',
                rf'ensure \w+ {lemma}',
            ])
    return patterns


# Generate expanded quality attributes and patterns
quality_attributes = generate_quality_attributes()
quality_patterns = generate_patterns(quality_attributes)


def analyze_documentation(doc_text):
    # Lemmatize the text
    lemmatized_text = lemmatize_text(doc_text)

    # Use spaCy for advanced NLP tasks
    doc = nlp(lemmatized_text)

    # Extract key phrases using noun chunks
    key_phrases = [chunk.text for chunk in doc.noun_chunks]

    # Perform topic modeling
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform([lemmatized_text])

    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(doc_term_matrix)

    # Get the top words for each topic
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
        topics.append(top_words)

    # Identify sections likely containing quality requirements
    quality_sections = []
    for sent in doc.sents:
        if any(keyword in sent.text.lower() for keyword in
               ['performance', 'security', 'maintainability', 'scalability']):
            quality_sections.append(sent.text)

    return {
        'key_phrases': key_phrases,
        'topics': topics,
        'quality_sections': quality_sections
    }


def analyze_issues_and_comments(issues_and_comments):
    classified_issues = {attr: [] for attr in quality_attributes.keys()}
    sentiments = {}

    for item in issues_and_comments:
        text = lemmatize_text(item['text'])

        # Classify based on expanded attributes
        for attr, words in quality_attributes.items():
            if any(lemmatizer.lemmatize(word) in text.lower() for word in words):
                classified_issues[attr].append(item)

        # Perform sentiment analysis
        blob = TextBlob(text)
        sentiments[item['id']] = blob.sentiment.polarity

    return classified_issues, sentiments


def extract_quality_requirements(text):
    lemmatized_text = lemmatize_text(text)
    quality_reqs = []
    for pattern in quality_patterns:
        matches = re.finditer(pattern, lemmatized_text, re.IGNORECASE)
        for match in matches:
            quality_reqs.append({
                'requirement': match.group(0),
                'start': match.start(),
                'end': match.end()
            })

    return quality_reqs


def identify_quality_attribute(requirement):
    lemmatized_req = lemmatize_text(requirement)
    for attr, words in quality_attributes.items():
        if any(lemmatizer.lemmatize(word) in lemmatized_req.lower() for word in words):
            return attr
    return 'unknown'


def analyze_quality_requirements(text):
    requirements = extract_quality_requirements(text)
    categorized_reqs = {}

    for req in requirements:
        attr = identify_quality_attribute(req['requirement'])
        if attr not in categorized_reqs:
            categorized_reqs[attr] = []
        categorized_reqs[attr].append(req)

    return categorized_reqs


attributes_test = {
    'sample': ['perf', "optimiz", "speed", "fast"]

}


def analyze_documentation2(docs: str):
    sentences = re.split(r'[\n.]', docs)
    reverse_index = {}
    # for attr, words in quality_attributes.items():
    for attr, words in attributes_test.items():
        reverse_index[attr] = {}
        for word in words:
            for sentence in sentences:
                pattern = re.compile(rf'\b{word}')
                match = re.search(pattern, sentence)
                if match:
                    if word in reverse_index[attr]:
                        reverse_index[attr][word].append(sentence)
                    else:
                        reverse_index[attr][word] = [sentence]
    return reverse_index


def strip_html_tags(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text()


# Example usage
if __name__ == "__main__":
    # Sample data (replace with actual data in a real scenario)
    # documentation = """
    # Our software aims to provide a lightning-fast user experience while ensuring top-notch security.
    # The system should adapt seamlessly to increasing user loads without compromising on performance.
    # We prioritize writing clean, well-documented code to facilitate long-term maintenance and evolution of the software.
    # The interface must be intuitive and accessible to users of all skill levels.
    # """
    #
    # issues_and_comments = [
    #     {'id': 1, 'text': "The dashboard takes forever to load. We need to optimize it for snappier response times."},
    #     {'id': 2,
    #      'text': "Users are reporting that the app feels sluggish on older devices. Let's improve its efficiency."},
    #     {'id': 3,
    #      'text': "We should implement proper encryption for all sensitive data transfers to bolster our security measures."},
    #     {'id': 4,
    #      'text': "The codebase is becoming unwieldy. We need to refactor the user management module for better maintainability."},
    #     {'id': 5,
    #      'text': "As we onboard more enterprise clients, we need to ensure our infrastructure can handle the increased load gracefully."}
    # ]

    matches = {}
    PATH = ".tmp/docs/Scanpy/scanpy.readthedocs.io/en"
    for root, dirs, files in os.walk(PATH):
        for file in files:
            if file.endswith(".html"):
                documentation_raw = open(os.path.join(root, file), "r", encoding="utf-8").read()
                text_content = strip_html_tags(documentation_raw)
                doc_analysis = analyze_documentation2(text_content)
                # print(f"Documentation Analysis for : {file}", doc_analysis)
                matches[file] = doc_analysis

    print(matches)

    # Analyze documentation

    # Analyze issues and comments
    # classified_issues, sentiments = analyze_issues_and_comments(issues_and_comments)
    # print("Classified Issues:", classified_issues)
    # print("Sentiments:", sentiments)
    #
    # # Extract and categorize quality requirements
    # all_text = documentation + " ".join([item['text'] for item in issues_and_comments])
    # categorized_quality_reqs = analyze_quality_requirements(all_text)
    # print("Categorized Quality Requirements:", categorized_quality_reqs)
