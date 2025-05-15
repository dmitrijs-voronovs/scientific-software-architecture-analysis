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
from nltk.stem import PorterStemmer
from nltk import WordNetLemmatizer, ngrams as nltk_ngrams
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.probability import FreqDist

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
from collections import defaultdict # For stem_to_raw_map

# --- Configuration ---
SEED_KEYWORDS_RAW = [
    "performance", "latency", "throughput", "response", "load", "scalability", "bottleneck", # Corrected scalab
    "availability", "downtime", "reliability", "fault", "failure", "resilience", "recovery", # Corrected reliab, resilien, recover
    "security", "attack", "vulnerability", "encryption", "authorization", "authentication", # Corrected vulnerab, encrypt, authoriz, authentcat
]

USE_STEMMING = True
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# --- Helper Functions ---

def extract_text_from_pdf(pdf_path):
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
    """
    Cleans and preprocesses text.
    Returns:
        - final_processed_tokens: list of stemmed/lemmatized tokens.
        - stem_to_raw_map: dict mapping processed_token -> set of original raw_tokens.
    """
    if not text:
        return [], defaultdict(set)

    original_text_for_mapping = text # Keep a copy before lowercasing for accurate raw word mapping

    text = re.sub(r'([a-z])-\n([a-z])', r'\1\2', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'-\n', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = text.lower()
    text_for_tokenization = re.sub(r'[^a-z\s-]', '', text)
    raw_tokens = nltk.word_tokenize(text_for_tokenization) # Tokenize before further cleaning for map

    processed_tokens_with_originals = [] # Stores tuples of (original_cleaned_token, processed_token)
    stem_to_raw_map = defaultdict(set)

    for raw_token_idx, token in enumerate(raw_tokens):
        original_cleaned_token = token.strip('-')

        if not original_cleaned_token or all(c == '-' for c in token):
            continue

        if original_cleaned_token not in stop_words and len(original_cleaned_token) > 2:
            if original_cleaned_token != '-': # Ensure it's not just a hyphen

                if USE_STEMMING:
                    processed_token = stemmer.stem(original_cleaned_token)
                else:
                    processed_token = lemmatizer.lemmatize(original_cleaned_token)

                # Add to map: processed_token -> original_cleaned_token
                # We use original_cleaned_token because it's the version before stemming/lemma
                # and after initial cleaning (hyphens, lowercase)
                stem_to_raw_map[processed_token].add(original_cleaned_token)
                processed_tokens_with_originals.append(processed_token)


    final_processed_tokens = processed_tokens_with_originals # This is now just the list of processed tokens
    # print(f"Preprocessing reduced text to {len(final_processed_tokens)} tokens.")
    # print(f"Stem-to-raw map contains {len(stem_to_raw_map)} entries.")
    return final_processed_tokens, stem_to_raw_map


def get_synonyms_wordnet(word, pos=None, max_synonyms_per_pos=3):
    synonyms = set()
    pos_tags_to_try = [wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADV]
    if pos:
        pos_tags_to_try = [pos]
    for wn_pos in pos_tags_to_try:
        synsets = wordnet.synsets(word, pos=wn_pos)
        count = 0
        for synset in synsets:
            for lemma in synset.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
                    count += 1
                    if count >= max_synonyms_per_pos: break
            if count >= max_synonyms_per_pos: break
    return list(synonyms)

def process_seed_keywords(raw_keywords, expand_synonyms=False, max_synonyms_per_seed=3):
    all_keywords_to_process = set(k.lower() for k in raw_keywords)
    if expand_synonyms:
        # print(f"Attempting to expand {len(raw_keywords)} raw seed keywords with synonyms...")
        expanded_keywords = set(all_keywords_to_process)
        for seed_word in raw_keywords:
            syns = get_synonyms_wordnet(seed_word.lower(), max_synonyms_per_pos=max_synonyms_per_seed)
            if syns:
                for syn in syns: expanded_keywords.add(syn.lower())
        # print(f"  Expanded to {len(expanded_keywords)} unique terms (including originals) before processing.")
        all_keywords_to_process = expanded_keywords
    # else:
        # print("Synonym expansion skipped.")

    if not all_keywords_to_process:
        # print("No keywords to process.")
        return set(), defaultdict(set) # Return empty map as well

    text_of_all_keywords = " ".join(all_keywords_to_process)
    # Since process_seed_keywords is about getting the *processed* seeds,
    # we primarily care about the first return value of preprocess_text.
    # The stem_to_raw_map for seeds isn't typically used downstream, but we can generate it.
    processed_seed_tokens, seed_stem_to_raw_map = preprocess_text(text_of_all_keywords)
    processed_seeds_set = set(processed_seed_tokens)

    # print(f"Processed {len(raw_keywords)} raw seed keywords (expanded to {len(all_keywords_to_process)} terms before processing) into {len(processed_seeds_set)} unique processed seed terms.")
    return processed_seeds_set # Only return the set of processed seeds

# --- Helper for formatting original words ---
def format_original_words(original_set):
    if not original_set:
        return ""
    if len(original_set) == 1:
        return list(original_set)[0]
    return f"({', '.join(sorted(list(original_set)))})"

# --- Analysis Functions ---
def calculate_tfidf(processed_tokens, stem_to_raw_map):
    if not processed_tokens:
        return pd.DataFrame(columns=['Term (Processed)', 'Original Word(s)', 'TF-IDF Score'])
    text_for_tfidf = [" ".join(processed_tokens)]
    vectorizer = TfidfVectorizer(ngram_range=(1, 1))
    try:
        tfidf_matrix = vectorizer.fit_transform(text_for_tfidf)
        feature_names = vectorizer.get_feature_names_out() # These are the processed terms
        scores = tfidf_matrix.toarray().flatten()

        df_tfidf = pd.DataFrame({'Term (Processed)': feature_names, 'TF-IDF Score': scores})
        df_tfidf['Original Word(s)'] = df_tfidf['Term (Processed)'].apply(lambda term: format_original_words(stem_to_raw_map.get(term, set())))
        df_tfidf = df_tfidf.sort_values(by='TF-IDF Score', ascending=False).reset_index(drop=True)
        df_tfidf = df_tfidf[['Term (Processed)', 'Original Word(s)', 'TF-IDF Score']] # Reorder
        # print(f"Calculated TF-IDF for {len(df_tfidf)} terms.")
        return df_tfidf
    except ValueError as e:
        # print(f"Could not calculate TF-IDF. Reason: {e}")
        return pd.DataFrame(columns=['Term (Processed)', 'Original Word(s)', 'TF-IDF Score'])

def extract_ngrams(processed_tokens, stem_to_raw_map, n=2, num_top_ngrams=50):
    if not processed_tokens or len(processed_tokens) < n:
        return pd.DataFrame(columns=[f'{n}-gram (Processed)', f'Original {n}-gram(s)', 'Frequency'])

    n_grams_tuples = list(nltk_ngrams(processed_tokens, n))
    if not n_grams_tuples:
         return pd.DataFrame(columns=[f'{n}-gram (Processed)', f'Original {n}-gram(s)', 'Frequency'])

    freq_dist = FreqDist(n_grams_tuples)
    most_common = freq_dist.most_common(num_top_ngrams)

    ngram_processed_list = []
    ngram_original_list = []
    freq_list = []

    for ngram_tuple, freq in most_common:
        ngram_processed_list.append(" ".join(ngram_tuple))
        # Reconstruct original n-gram possibilities
        # This can get complex if each part has multiple originals. For simplicity,
        # we'll show one common combination or list them per part.
        # Let's show the most common form based on map, or list all if too complex
        original_parts_list = []
        for token_in_ngram in ngram_tuple:
            originals = stem_to_raw_map.get(token_in_ngram, {token_in_ngram}) # Fallback to processed if not in map
            original_parts_list.append(format_original_words(originals))
        ngram_original_list.append(" ".join(original_parts_list))
        freq_list.append(freq)

    df_ngrams = pd.DataFrame({
        f'{n}-gram (Processed)': ngram_processed_list,
        f'Original {n}-gram(s)': ngram_original_list,
        'Frequency': freq_list
    })
    # print(f"Extracted top {len(df_ngrams)} {n}-grams.")
    return df_ngrams

def find_collocations_general(processed_tokens, stem_to_raw_map, num_collocations=50, window_size=5):
    """Finds general word collocations (bigrams) based on PMI."""
    if not processed_tokens:
         return pd.DataFrame(columns=['Collocation (Processed)', 'Original Collocation(s)', 'PMI Score'])

    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(processed_tokens, window_size=window_size)
    # finder.apply_freq_filter(2) # Optional: filter low-frequency pairs

    try:
        scored = finder.score_ngrams(bigram_measures.pmi)
    except ZeroDivisionError: # Can happen if all words are unique or very sparse
        print("Warning: ZeroDivisionError in collocation scoring (likely sparse data). Returning empty DataFrame.")
        return pd.DataFrame(columns=['Collocation (Processed)', 'Original Collocation(s)', 'PMI Score'])


    if not scored:
        # print("No general collocations found.")
        return pd.DataFrame(columns=['Collocation (Processed)', 'Original Collocation(s)', 'PMI Score'])

    processed_collocs = []
    original_collocs = []
    pmi_scores = []

    for (w1_proc, w2_proc), score in scored:
        if len(processed_collocs) >= num_collocations:
            break
        processed_collocs.append(f"{w1_proc} {w2_proc}")
        orig_w1 = format_original_words(stem_to_raw_map.get(w1_proc, {w1_proc}))
        orig_w2 = format_original_words(stem_to_raw_map.get(w2_proc, {w2_proc}))
        original_collocs.append(f"{orig_w1} {orig_w2}")
        pmi_scores.append(score)

    df_collocations = pd.DataFrame({
        'Collocation (Processed)': processed_collocs,
        'Original Collocation(s)': original_collocs,
        'PMI Score': pmi_scores
    })
    df_collocations = df_collocations.sort_values(by='PMI Score', ascending=False).reset_index(drop=True)
    # print(f"Found top {len(df_collocations)} general collocations (PMI based).")
    return df_collocations.head(num_collocations)


def find_collocations_with_seeds(processed_tokens, stem_to_raw_map, processed_seed_keywords, num_collocations=50, window_size=5):
    if not processed_tokens or not processed_seed_keywords:
         return pd.DataFrame(columns=['Collocation (Processed)', 'Original Collocation(s)', 'PMI Score', 'Contains Seed'])
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(processed_tokens, window_size=window_size)
    # finder.apply_freq_filter(2)

    try:
        scored = finder.score_ngrams(bigram_measures.pmi)
    except ZeroDivisionError:
        print("Warning: ZeroDivisionError in seed collocation scoring (likely sparse data). Returning empty DataFrame.")
        return pd.DataFrame(columns=['Collocation (Processed)', 'Original Collocation(s)', 'PMI Score', 'Contains Seed'])


    collocations_data = []
    for (w1_proc, w2_proc), score in scored:
        if w1_proc in processed_seed_keywords or w2_proc in processed_seed_keywords:
            orig_w1 = format_original_words(stem_to_raw_map.get(w1_proc, {w1_proc}))
            orig_w2 = format_original_words(stem_to_raw_map.get(w2_proc, {w2_proc}))
            collocations_data.append({
                'Collocation (Processed)': f"{w1_proc} {w2_proc}",
                'Original Collocation(s)': f"{orig_w1} {orig_w2}",
                'PMI Score': score,
                'Contains Seed': True
            })

    if not collocations_data:
        # print("No collocations found involving the seed keywords.")
        return pd.DataFrame(columns=['Collocation (Processed)', 'Original Collocation(s)', 'PMI Score', 'Contains Seed'])

    df_collocations = pd.DataFrame(collocations_data)
    df_collocations = df_collocations.sort_values(by='PMI Score', ascending=False).reset_index(drop=True)
    # print(f"Found {len(df_collocations)} collocations involving seed keywords (PMI based).")
    return df_collocations.head(num_collocations)


def main():
    PDF_PATH = "metadata/papers/security/wellarchitected-security-pillar.pdf"  # <--- CHANGE THIS TO YOUR PDF FILE PATH
    OUTPUT_EXCEL = "keyword_analysis_results_final.xlsx"
    NUM_TOP_BIGRAMS = 100
    NUM_TOP_TRIGRAMS = 50
    NUM_COLLOCATIONS_GENERAL = 100
    NUM_COLLOCATIONS_SEEDS = 100
    COLLOCATION_WINDOW_SIZE = 5
    EXPAND_SEED_SYNONYMS = False # Set to True to expand seed keywords with synonyms

    print("Starting keyword analysis...")
    raw_text = extract_text_from_pdf(PDF_PATH)

    if raw_text:
        processed_tokens, stem_to_raw_map = preprocess_text(raw_text)
        print(f"Preprocessing completed. Total processed tokens: {len(processed_tokens)}. Stem-to-raw map size: {len(stem_to_raw_map)}.")

        processed_seeds = process_seed_keywords(SEED_KEYWORDS_RAW, expand_synonyms=EXPAND_SEED_SYNONYMS)
        print(f"Seed keywords processed. Total unique processed seeds: {len(processed_seeds)}.")


        print("\n--- Running Analyses ---")
        df_tfidf = calculate_tfidf(processed_tokens, stem_to_raw_map)
        df_bigrams = extract_ngrams(processed_tokens, stem_to_raw_map, n=2, num_top_ngrams=NUM_TOP_BIGRAMS)
        df_trigrams = extract_ngrams(processed_tokens, stem_to_raw_map, n=3, num_top_ngrams=NUM_TOP_TRIGRAMS)

        df_collocations_general = find_collocations_general(
            processed_tokens, stem_to_raw_map,
            num_collocations=NUM_COLLOCATIONS_GENERAL,
            window_size=COLLOCATION_WINDOW_SIZE
        )
        df_collocations_seeds = find_collocations_with_seeds(
            processed_tokens, stem_to_raw_map, processed_seeds,
            num_collocations=NUM_COLLOCATIONS_SEEDS,
            window_size=COLLOCATION_WINDOW_SIZE
        )

        print(f"\n--- Saving results to {OUTPUT_EXCEL} ---")
        try:
            with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
                if not df_tfidf.empty: df_tfidf.to_excel(writer, sheet_name='TF-IDF', index=False)
                if not df_bigrams.empty: df_bigrams.to_excel(writer, sheet_name='Top Bigrams', index=False)
                if not df_trigrams.empty: df_trigrams.to_excel(writer, sheet_name='Top Trigrams', index=False)
                if not df_collocations_general.empty: df_collocations_general.to_excel(writer, sheet_name='Collocations (General)', index=False)
                if not df_collocations_seeds.empty: df_collocations_seeds.to_excel(writer, sheet_name='Collocations (with Seeds)', index=False)
            print("Successfully saved results to Excel.")
        except Exception as e:
            print(f"Error saving results to Excel: {e}")
            # Fallback to CSV can be added here if needed
    else:
        print("Could not extract text from PDF. Exiting.")
    print("\nAnalysis finished.")

if __name__ == "__main__":
    # Optional: Download NLTK resources if not present
    # try: nltk.data.find('corpora/wordnet')
    # except LookupError: nltk.download('wordnet')
    # try: nltk.data.find('corpora/stopwords')
    # except LookupError: nltk.download('stopwords')
    # try: nltk.data.find('tokenizers/punkt')
    # except LookupError: nltk.download('punkt')
    main()