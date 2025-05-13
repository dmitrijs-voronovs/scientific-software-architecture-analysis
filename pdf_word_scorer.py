# import nltk
# try:
#     nltk.data.find('corpora/wordnet')
# except nltk.downloader.DownloadError:
#     nltk.download('wordnet')
# try:
#     nltk.data.find('corpora/stopwords')
# except nltk.downloader.DownloadError:
#     nltk.download('stopwords')
# try:
#     nltk.data.find('tokenizers/punkt')
# except nltk.downloader.DownloadError:
#     nltk.download('punkt')

import PyPDF2
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer # Using PorterStemmer as in the example
# from nltk.stem import WordNetLemmatizer # Alternative: Lemmatization
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.probability import FreqDist
from nltk import ngrams as nltk_ngrams, WordNetLemmatizer  # Avoid name clash with our function

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
import os
from collections import Counter

# --- Configuration ---
# Define seed keywords relevant to your QAs (use stemmed/lemmatized form if preprocessing includes it)
# Example seed keywords (remember to stem/lemmatize them if your preprocessing does)
SEED_KEYWORDS_RAW = [
    "perform", "latency", "throughput", "response", "load", "scalab", "bottleneck",
    "avail", "downtime", "reliab", "fault", "failure", "resilien", "recover",
    "secur", "attack", "vulnerab", "encrypt", "authoriz", "authenticat",
    # Add more keywords relevant to your quality attributes
]

# Choose stemming or lemmatization
USE_STEMMING = True # Set to False to use Lemmatization instead
lemmatizer = WordNetLemmatizer() # Uncomment if USE_STEMMING = False
stemmer = PorterStemmer()         # Used if USE_STEMMING = True

stop_words = set(stopwords.words('english'))

# --- Helper Functions ---

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            print(f"Reading {num_pages} pages from {pdf_path}...")
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    print(f"Warning: Could not extract text from page {page_num + 1}. Error: {e}")
        print(f"Successfully extracted text (length: {len(text)} characters).")
        return text
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        return None
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return None

def preprocess_text(text):
    # 1. Handle line-ending hyphens (de-hyphenation)
    #    Looks for a letter, a hyphen, a newline, and then another letter.
    #    Replaces this pattern by joining the letters without the hyphen and newline.
    text = re.sub(r'([a-z])-\n([a-z])', r'\1\2', text, flags=re.IGNORECASE | re.MULTILINE)
    #    Then, any remaining hyphens at the very end of a line followed by newline can be just removed
    #    (e.g. if a word was hyphenated but the next line was empty or started with non-alpha)
    text = re.sub(r'-\n', '', text, flags=re.IGNORECASE | re.MULTILINE)


    # 2. Lowercase
    text = text.lower()

    # 3. Preserve intra-word hyphens, remove other punctuation.
    #    Keeps lowercase letters, spaces, and hyphens that are part of words.
    #    A hyphen is considered "part of a word" if it's surrounded by letters,
    #    or at the start/end of a word if it forms part of it (e.g., "e-mail").
    #    This regex:
    #    - keeps a-z
    #    - keeps spaces \s
    #    - keeps hyphens -
    #    Then we'll clean up hyphens that are not part of words after tokenization.
    text = re.sub(r'[^a-z\s-]', '', text) # Keep letters, spaces, and hyphens for now

    # 4. Tokenize
    #    NLTK's word_tokenize is generally good at handling hyphenated words like "state-of-the-art" as single tokens.
    tokens = nltk.word_tokenize(text)

    # 5. Post-tokenization hyphen cleaning and general processing
    processed_tokens = []
    for token in tokens:
        # Remove leading/trailing hyphens that are not part of a valid word structure
        # e.g. " -word- " becomes "word"
        # A simple check: if a token is just hyphens, or starts/ends with one and isn't a valid hyphenated word
        cleaned_token = token.strip('-')

        # If stripping hyphens made the token empty, or if the original token was just hyphens, skip
        if not cleaned_token or all(c == '-' for c in token):
            continue

        # Check if the token should be kept (not a stop word, long enough)
        if cleaned_token not in stop_words and len(cleaned_token) > 2:
            # Ensure the token itself isn't just a hyphen after cleaning (though previous check should catch this)
            # and isn't a stop word after cleaning.
             if cleaned_token != '-' and cleaned_token not in stop_words: # Redundant check for stop_words, but safe
                processed_tokens.append(cleaned_token)


    # 6. Apply Stemming or Lemmatization
    final_tokens = []
    if USE_STEMMING:
        final_tokens = [stemmer.stem(word) for word in processed_tokens]
    else: # Assuming Lemmatization
        # final_tokens = [lemmatizer.lemmatize(word) for word in processed_tokens] # Make sure lemmatizer is defined
        final_tokens = processed_tokens # Fallback if lemmatizer not set up for this example
        if 'lemmatizer' in globals() or 'lemmatizer' in locals():
             final_tokens = [lemmatizer.lemmatize(word) for word in processed_tokens]


    # print(f"Preprocessing reduced text to {len(final_tokens)} tokens.") # Moved print for clarity
    return final_tokens


def get_synonyms_wordnet(word, pos=None, max_synonyms_per_pos=3):
    """
    Fetches synonyms for a given word using WordNet.
    Tries common POS tags if 'pos' is None.
    """
    synonyms = set()
    # Define WordNet POS tags to try if specific POS is not given
    pos_tags_to_try = [wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADV]
    if pos: # If a specific POS tag is provided (e.g., wordnet.NOUN)
        pos_tags_to_try = [pos]

    for wn_pos in pos_tags_to_try:
        synsets = wordnet.synsets(word, pos=wn_pos)
        count = 0
        for synset in synsets:
            for lemma in synset.lemmas():
                synonym = lemma.name().replace('_', ' ') # Replace underscores with spaces for multi-word synonyms
                if synonym.lower() != word.lower(): # Don't add the word itself
                    synonyms.add(synonym)
                    count += 1
                    if count >= max_synonyms_per_pos:
                        break
            if count >= max_synonyms_per_pos:
                break
    return list(synonyms)

def process_seed_keywords(raw_keywords, expand_synonyms=False, max_synonyms_per_seed=3):
    """
    Processes seed keywords:
    1. Expands the list with synonyms (if expand_synonyms is True).
    2. Applies the same preprocessing (stemming/lemmatization) to all keywords.
    """
    all_keywords_to_process = set(k.lower() for k in raw_keywords) # Start with original keywords, lowercased

    if expand_synonyms:
        print(f"Attempting to expand {len(raw_keywords)} raw seed keywords with synonyms...")
        expanded_keywords = set(all_keywords_to_process) # Use a new set for expansion
        for seed_word in raw_keywords: # Iterate over original raw keywords for synonym lookup
            syns = get_synonyms_wordnet(seed_word.lower(), max_synonyms_per_pos=max_synonyms_per_seed)
            if syns:
                # print(f"  Found synonyms for '{seed_word}': {syns}")
                for syn in syns:
                    expanded_keywords.add(syn.lower()) # Add synonyms in lowercase
        print(f"  Expanded to {len(expanded_keywords)} unique terms (including originals) before processing.")
        all_keywords_to_process = expanded_keywords
    else:
        print("Synonym expansion skipped.")

    # Now, preprocess ALL keywords (original + synonyms if any)
    # Your original method of joining and then preprocessing works well here.
    # This ensures multi-word synonyms are tokenized and processed correctly.
    if not all_keywords_to_process:
        print("No keywords to process.")
        return set()

    text_of_all_keywords = " ".join(all_keywords_to_process)
    processed_seeds = set(preprocess_text(text_of_all_keywords)) # preprocess_text handles tokenization, stemming/lemmatizing

    print(f"Processed {len(raw_keywords)} raw seed keywords (expanded to {len(all_keywords_to_process)} terms before processing) into {len(processed_seeds)} unique processed seed terms.")
    return processed_seeds


# --- Analysis Functions ---

def calculate_tfidf(processed_tokens):
    """
    Calculates TF-IDF scores for tokens in the document.
    Note: For a single document, IDF is constant, so this reflects term frequency adjusted
          by how common the term is *if* more documents were present in the fit.
          For finding important terms *within* a single doc, raw TF might be simpler,
          but this shows the TF-IDF mechanic.
    """
    if not processed_tokens:
        return pd.DataFrame(columns=['Term', 'TF-IDF Score'])

    # TF-IDF requires strings; join tokens back. Treat the single doc as a corpus of one.
    text_for_tfidf = [" ".join(processed_tokens)]

    vectorizer = TfidfVectorizer(ngram_range=(1, 1)) # Use (1, 2) to include bigrams
    try:
        tfidf_matrix = vectorizer.fit_transform(text_for_tfidf)
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray().flatten()

        # Create DataFrame
        df_tfidf = pd.DataFrame({'Term': feature_names, 'TF-IDF Score': scores})
        df_tfidf = df_tfidf.sort_values(by='TF-IDF Score', ascending=False).reset_index(drop=True)
        print(f"Calculated TF-IDF for {len(df_tfidf)} terms.")
        return df_tfidf

    except ValueError as e:
        # Handle case where vocabulary might be empty after processing
        print(f"Could not calculate TF-IDF. Reason: {e}")
        if "empty vocabulary" in str(e):
            print("This often happens if the text contains only stopwords or very short words after preprocessing.")
        return pd.DataFrame(columns=['Term', 'TF-IDF Score'])


def extract_ngrams(processed_tokens, n=2, num_top_ngrams=50):
    """Extracts most frequent N-grams."""
    if not processed_tokens or len(processed_tokens) < n:
        return pd.DataFrame(columns=[f'{n}-gram', 'Frequency'])

    n_grams = list(nltk_ngrams(processed_tokens, n))
    if not n_grams:
         return pd.DataFrame(columns=[f'{n}-gram', 'Frequency'])

    freq_dist = FreqDist(n_grams)
    most_common = freq_dist.most_common(num_top_ngrams)

    # Format for DataFrame
    ngram_list = [" ".join(ngram) for ngram, freq in most_common]
    freq_list = [freq for ngram, freq in most_common]

    df_ngrams = pd.DataFrame({f'{n}-gram': ngram_list, 'Frequency': freq_list})
    print(f"Extracted top {len(df_ngrams)} {n}-grams.")
    return df_ngrams


def find_collocations(processed_tokens, processed_seed_keywords, num_collocations=50, window_size=5):
    """
    Finds words that frequently co-occur with seed keywords within a window.
    Uses Pointwise Mutual Information (PMI) to score collocations (bigrams).
    Filters results to show pairs containing at least one seed keyword.
    """
    if not processed_tokens or not processed_seed_keywords:
         return pd.DataFrame(columns=['Collocation (Bigram)', 'PMI Score', 'Contains Seed'])

    bigram_measures = BigramAssocMeasures()
    # Consider words within 'window_size' distance
    finder = BigramCollocationFinder.from_words(processed_tokens, window_size=window_size)

    # Optional: Filter out low-frequency words/pairs early to speed up
    # finder.apply_freq_filter(3)

    # Score based on PMI
    # Other measures exist: bigram_measures.raw_freq, bigram_measures.chi_sq, etc.
    scored = finder.score_ngrams(bigram_measures.pmi)

    # Filter for collocations involving at least one seed keyword
    collocations_with_seeds = []
    for (w1, w2), score in scored:
        if w1 in processed_seed_keywords or w2 in processed_seed_keywords:
            collocations_with_seeds.append( (f"{w1} {w2}", score, True) )

    # Create DataFrame
    if not collocations_with_seeds:
        print("No collocations found involving the seed keywords.")
        return pd.DataFrame(columns=['Collocation (Bigram)', 'PMI Score', 'Contains Seed'])

    df_collocations = pd.DataFrame(collocations_with_seeds, columns=['Collocation (Bigram)', 'PMI Score', 'Contains Seed'])
    df_collocations = df_collocations.sort_values(by='PMI Score', ascending=False).reset_index(drop=True)

    # Limit to top N
    df_collocations = df_collocations.head(num_collocations)
    print(f"Found {len(df_collocations)} collocations involving seed keywords (PMI based).")
    return df_collocations

def main():
    # --- Configuration ---
    PDF_PATH = "metadata/papers/security/wellarchitected-security-pillar.pdf"  # <--- CHANGE THIS TO YOUR PDF FILE PATH
    OUTPUT_EXCEL = "keyword_analysis_results2.xlsx"
    NUM_TOP_BIGRAMS = 100
    NUM_TOP_TRIGRAMS = 50
    NUM_COLLOCATIONS = 100
    COLLOCATION_WINDOW_SIZE = 5  # How many words apart can collocating words be?

    print("Starting keyword analysis...")

    # 1. Extract Text
    raw_text = extract_text_from_pdf(PDF_PATH)

    if raw_text:
        # 2. Preprocess Text
        processed_seeds = process_seed_keywords(SEED_KEYWORDS_RAW)
        processed_tokens = preprocess_text(raw_text)

        # 3. Run Analyses
        print("\n--- Running Analyses ---")
        df_tfidf = calculate_tfidf(processed_tokens)
        df_bigrams = extract_ngrams(processed_tokens, n=2, num_top_ngrams=NUM_TOP_BIGRAMS)
        df_trigrams = extract_ngrams(processed_tokens, n=3, num_top_ngrams=NUM_TOP_TRIGRAMS)
        df_collocations = find_collocations(processed_tokens, processed_seeds,
                                            num_collocations=NUM_COLLOCATIONS,
                                            window_size=COLLOCATION_WINDOW_SIZE)

        # 4. Save to Excel
        print(f"\n--- Saving results to {OUTPUT_EXCEL} ---")
        try:
            with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
                df_tfidf.to_excel(writer, sheet_name='TF-IDF', index=False)
                df_bigrams.to_excel(writer, sheet_name='Top Bigrams', index=False)
                df_trigrams.to_excel(writer, sheet_name='Top Trigrams', index=False)
                df_collocations.to_excel(writer, sheet_name='Collocations (with Seeds)', index=False)
            print("Successfully saved results to Excel.")
        except Exception as e:
            print(f"Error saving results to Excel: {e}")
            print("Attempting to save as CSV files instead...")
            try:
                df_tfidf.to_csv("tfidf_results.csv", index=False)
                df_bigrams.to_csv("bigram_results.csv", index=False)
                df_trigrams.to_csv("trigram_results.csv", index=False)
                df_collocations.to_csv("collocation_results.csv", index=False)
                print("Successfully saved results as separate CSV files.")
            except Exception as csve:
                print(f"Could not save as CSV either: {csve}")


    else:
        print("Could not extract text from PDF. Exiting.")

    print("\nAnalysis finished.")

# --- Main Execution ---
if __name__ == "__main__":
    main()