import os
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
from collections import defaultdict
import shelve  # For caching
import hashlib  # For content hashing
from pathlib import Path  # For easier path manipulation

from quality_attributes import quality_attributes  # Assuming this file exists and contains the map

# --- Configuration ---
# SEED_WORDS_RAW_MAP will be loaded from quality_attributes.py
# Example:
# quality_attributes_map = {
#     "Security": ["security", "attack", ...],
#     "Performance": ["performance", "latency", ...],
# }
SEED_WORDS_RAW_MAP = quality_attributes

USE_STEMMING = True  # True for Stemming, False for Lemmatization
PREPROCESSING_VERSION = "v1.1"  # Increment if preprocess_text logic changes significantly

# --- Caching Configuration ---
CACHE_DIR = ".cache/pdf_cache"
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)  # Ensure cache directory exists


# --- NLTK Resource Download (run once if needed) ---
def ensure_nltk_resources():
    resources = [('corpora/wordnet', 'wordnet'),
                 ('corpora/stopwords', 'stopwords'),
                 ('tokenizers/punkt', 'punkt')]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except nltk.downloader.DownloadError:
            print(f"Downloading NLTK resource: {name}")
            nltk.download(name)


# --- Global NLP Objects (initialize once) ---
if USE_STEMMING:
    stemmer = PorterStemmer()
else:
    lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# --- Helper Functions (find_pdfs_by_qa, extract_text_from_pdf, format_original_words, get_synonyms_wordnet are mostly the same) ---
def find_pdfs_by_qa(base_dir):
    qa_to_pdfs = defaultdict(list)
    if not os.path.isdir(base_dir):
        print(f"Error: Base PDF directory '{base_dir}' not found.")
        return qa_to_pdfs

    for dir_item_name in os.listdir(base_dir):
        item_path = os.path.join(base_dir, dir_item_name)
        if os.path.isdir(item_path):
            # Match directory name (potential QA name) to SEED_WORDS_RAW_MAP keys
            # We need to find the original casing of the key from SEED_WORDS_RAW_MAP
            matched_qa_key = next((key_in_map for key_in_map in SEED_WORDS_RAW_MAP.keys() if key_in_map.lower() == dir_item_name.lower()), None)

            if matched_qa_key:
                print(f"Found QA directory: {dir_item_name} (mapped to key: '{matched_qa_key}')")
                for root, _, files in os.walk(item_path):
                    for file in files:
                        if file.lower().endswith(".pdf"):
                            pdf_path = os.path.join(root, file)
                            qa_to_pdfs[matched_qa_key].append(pdf_path)  # Use the matched_qa_key
            else:
                print(f"Skipping directory '{dir_item_name}' as it's not a recognized QA in SEED_WORDS_RAW_MAP.")
    return qa_to_pdfs


def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            # print(f"Reading {num_pages} pages from {pdf_path}...") # Reduced verbosity
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception:  # pylint: disable=broad-except
                    # print(f"Warning: Could not extract text from page {page_num + 1} of {pdf_path}. Error: {e}")
                    pass  # Be less verbose for errors here
        # print(f"Successfully extracted text (length: {len(text)} characters) from {Path(pdf_path).name}.")
        return text
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        return None
    except Exception as e:  # pylint: disable=broad-except
        print(f"Error reading PDF {pdf_path}: {e}")
        return None


def get_file_hash(filepath):
    """Computes MD5 hash of a file."""
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            buf = f.read(65536)  # Read in 64k chunks
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    except FileNotFoundError:
        return None


def preprocess_text_and_map(text, pdf_path_for_log=""):
    """
    Cleans and preprocesses text for a single document.
    Returns:
        - final_processed_tokens: list of stemmed/lemmatized tokens.
        - stem_to_raw_map: dict mapping processed_token -> set of original raw_tokens from this doc.
    """
    if not text:
        return [], defaultdict(set)

    # Line-ending hyphens
    text = re.sub(r'([a-z])-\n([a-z])', r'\1\2', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'-\n', '', text, flags=re.IGNORECASE | re.MULTILINE)

    text_lower = text.lower()
    # Keep letters, spaces, and hyphens within words. Remove other punctuation/numbers.
    text_for_tokenization = re.sub(r'[^a-z\s-]', '', text_lower)

    raw_tokens = nltk.word_tokenize(text_for_tokenization)

    final_processed_tokens = []
    stem_to_raw_map = defaultdict(set)

    for token in raw_tokens:
        original_cleaned_token = token.strip('-')

        if not original_cleaned_token or all(c == '-' for c in token) or \
                original_cleaned_token in stop_words or len(original_cleaned_token) <= 2:
            continue

        if USE_STEMMING:
            processed_token = stemmer.stem(original_cleaned_token)
        else:
            processed_token = lemmatizer.lemmatize(original_cleaned_token)

        final_processed_tokens.append(processed_token)
        stem_to_raw_map[processed_token].add(original_cleaned_token)

    # if pdf_path_for_log: # Optional: for debugging
    #     print(f"Preprocessed {Path(pdf_path_for_log).name}: {len(final_processed_tokens)} tokens, map size {len(stem_to_raw_map)}")
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
        expanded_keywords = set(all_keywords_to_process)
        for seed_word in raw_keywords:
            syns = get_synonyms_wordnet(seed_word.lower(), max_synonyms_per_pos=max_synonyms_per_seed)
            if syns:
                for syn in syns: expanded_keywords.add(syn.lower())
        all_keywords_to_process = expanded_keywords

    if not all_keywords_to_process:
        return set()

    text_of_all_keywords = " ".join(all_keywords_to_process)
    # Use preprocess_text_and_map, but we only need the processed tokens for seeds
    processed_seed_tokens, _ = preprocess_text_and_map(text_of_all_keywords)
    return set(processed_seed_tokens)


def format_original_words(original_set):
    if not original_set: return ""
    if len(original_set) == 1: return list(original_set)[0]
    return f"({', '.join(sorted(list(original_set)))})"


# --- Analysis Functions (calculate_corpus_tfidf, extract_ngrams, find_collocations_general, find_collocations_with_seeds) ---
# These are largely the same as your last version, ensure they use the passed stem_to_raw_map correctly.
# The calculate_corpus_tfidf was already designed for corpus-wide IDF.

def calculate_corpus_tfidf(all_processed_docs_texts, all_stem_to_raw_maps_list, target_doc_indices,
                           fitted_vectorizer=None):
    if not all_processed_docs_texts:
        return [], None  # Return empty list and None for vectorizer if no docs

    if fitted_vectorizer is None:
        vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=2)  # min_df to avoid rare terms dominating IDF too much
        print(f"Fitting TF-IDF vectorizer on {len(all_processed_docs_texts)} documents from the entire corpus...")
        try:
            corpus_tfidf_matrix = vectorizer.fit_transform(all_processed_docs_texts)
            fitted_vectorizer = vectorizer  # Store the fitted vectorizer
        except ValueError as e:
            print(f"Could not fit TF-IDF vectorizer. Reason: {e}")
            return [pd.DataFrame(columns=['Term (Processed)', 'Original Word(s)', 'TF-IDF Score']) for _ in
                    target_doc_indices], None
    else:
        # Use the pre-fitted vectorizer
        print("Using pre-fitted TF-IDF vectorizer...")
        vectorizer = fitted_vectorizer
        try:
            corpus_tfidf_matrix = vectorizer.transform(all_processed_docs_texts)  # Just transform
        except ValueError as e:  # Should not happen if vectorizer was fit on compatible data
            print(f"Error transforming with pre-fitted TF-IDF vectorizer: {e}")
            return [pd.DataFrame(columns=['Term (Processed)', 'Original Word(s)', 'TF-IDF Score']) for _ in
                    target_doc_indices], fitted_vectorizer

    feature_names = vectorizer.get_feature_names_out()
    results_dfs = []

    for i, doc_idx in enumerate(target_doc_indices):  # Iterate only over target_doc_indices for output
        if doc_idx >= corpus_tfidf_matrix.shape[0]:
            print(f"Warning: Document index {doc_idx} out of bounds for TF-IDF matrix. Skipping.")
            results_dfs.append(pd.DataFrame(columns=['Term (Processed)', 'Original Word(s)', 'TF-IDF Score']))
            continue

        doc_vector = corpus_tfidf_matrix[doc_idx]
        term_scores = {feature_names[col_idx]: doc_vector[0, col_idx] for col_idx in doc_vector.nonzero()[1]}

        if not term_scores:
            results_dfs.append(pd.DataFrame(columns=['Term (Processed)', 'Original Word(s)', 'TF-IDF Score']))
            continue

        current_doc_stem_to_raw_map = all_stem_to_raw_maps_list[doc_idx]
        df_tfidf = pd.DataFrame(list(term_scores.items()), columns=['Term (Processed)', 'TF-IDF Score'])
        df_tfidf['Original Word(s)'] = df_tfidf['Term (Processed)'].apply(
            lambda term: format_original_words(current_doc_stem_to_raw_map.get(term, {term}))  # Fallback to term itself
        )
        df_tfidf = df_tfidf.sort_values(by='TF-IDF Score', ascending=False).reset_index(drop=True)
        df_tfidf = df_tfidf[['Term (Processed)', 'Original Word(s)', 'TF-IDF Score']]
        results_dfs.append(df_tfidf)

    return results_dfs, fitted_vectorizer


def extract_ngrams(processed_tokens, stem_to_raw_map, n=2, num_top_ngrams=50):
    # (No significant changes needed from your provided code, assuming inputs are correct)
    if not processed_tokens or len(processed_tokens) < n:
        return pd.DataFrame(columns=[f'{n}-gram (Processed)', f'Original {n}-gram(s)', 'Frequency'])
    n_grams_tuples = list(nltk_ngrams(processed_tokens, n))
    if not n_grams_tuples:
        return pd.DataFrame(columns=[f'{n}-gram (Processed)', f'Original {n}-gram(s)', 'Frequency'])
    freq_dist = FreqDist(n_grams_tuples)
    most_common = freq_dist.most_common(num_top_ngrams)
    ngram_processed_list, ngram_original_list, freq_list = [], [], []
    for ngram_tuple, freq in most_common:
        ngram_processed_list.append(" ".join(ngram_tuple))
        original_parts_list = [format_original_words(stem_to_raw_map.get(token, {token})) for token in ngram_tuple]
        ngram_original_list.append(" ".join(original_parts_list))
        freq_list.append(freq)
    return pd.DataFrame({
        f'{n}-gram (Processed)': ngram_processed_list,
        f'Original {n}-gram(s)': ngram_original_list,
        'Frequency': freq_list
    })


def find_collocations_general(processed_tokens, stem_to_raw_map, num_collocations=50, window_size=5):
    # (No significant changes needed from your provided code)
    if not processed_tokens or len(processed_tokens) < window_size:  # Check if enough tokens for finder
        return pd.DataFrame(columns=['Collocation (Processed)', 'Original Collocation(s)', 'PMI Score'])
    bigram_measures = BigramAssocMeasures()
    try:
        finder = BigramCollocationFinder.from_words(processed_tokens, window_size=window_size)
        # finder.apply_freq_filter(2) # Optional
        scored = finder.score_ngrams(bigram_measures.pmi)
    except (ValueError, ZeroDivisionError) as e:  # ValueError if not enough words for window
        print(f"Warning: Error in general collocation scoring ({e}). Returning empty DataFrame.")
        return pd.DataFrame(columns=['Collocation (Processed)', 'Original Collocation(s)', 'PMI Score'])
    if not scored:
        return pd.DataFrame(columns=['Collocation (Processed)', 'Original Collocation(s)', 'PMI Score'])
    processed_collocs, original_collocs, pmi_scores = [], [], []
    for (w1_proc, w2_proc), score in scored:
        if len(processed_collocs) >= num_collocations: break
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
    return df_collocations.sort_values(by='PMI Score', ascending=False).reset_index(drop=True).head(num_collocations)


def find_collocations_with_seeds(processed_tokens, stem_to_raw_map, processed_seed_keywords, num_collocations=50,
                                 window_size=5):
    # (No significant changes needed from your provided code)
    if not processed_tokens or not processed_seed_keywords or len(processed_tokens) < window_size:
        return pd.DataFrame(
            columns=['Collocation (Processed)', 'Original Collocation(s)', 'PMI Score', 'Contains Seed'])
    bigram_measures = BigramAssocMeasures()
    try:
        finder = BigramCollocationFinder.from_words(processed_tokens, window_size=window_size)
        scored = finder.score_ngrams(bigram_measures.pmi)
    except (ValueError, ZeroDivisionError) as e:
        print(f"Warning: Error in seed collocation scoring ({e}). Returning empty DataFrame.")
        return pd.DataFrame(
            columns=['Collocation (Processed)', 'Original Collocation(s)', 'PMI Score', 'Contains Seed'])

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
        return pd.DataFrame(
            columns=['Collocation (Processed)', 'Original Collocation(s)', 'PMI Score', 'Contains Seed'])
    df_collocations = pd.DataFrame(collocations_data)
    return df_collocations.sort_values(by='PMI Score', ascending=False).reset_index(drop=True).head(num_collocations)


# --- Main Execution Logic ---
def main():
    # ensure_nltk_resources()  # Ensure NLTK resources are downloaded

    BASE_PDF_DIR = "metadata/papers/"  # Main directory containing QA subfolders
    NUM_TOP_BIGRAMS = 100
    NUM_TOP_TRIGRAMS = 50
    NUM_COLLOCATIONS_GENERAL = 200
    NUM_COLLOCATIONS_SEEDS = 100
    COLLOCATION_WINDOW_SIZE = 5
    EXPAND_SEED_SYNONYMS = False
    # Optional: Set to True to force reprocessing of all files, ignoring cache
    FORCE_REPROCESS_ALL = False
    # Optional: Set to True to re-fit TF-IDF vectorizer even if cached preprocessed data is used
    FORCE_REFIT_TFIDF = False

    print("Starting CORPUS keyword analysis...")
    qa_to_pdf_paths_map = find_pdfs_by_qa(BASE_PDF_DIR)

    if not qa_to_pdf_paths_map:
        print("No PDFs found or QA directories configured. Exiting.")
        return

    # --- Phase 0: Cache Management and Corpus Ingestion ---
    all_docs_processed_tokens_list = []
    all_docs_stem_to_raw_maps_list = []
    all_docs_original_paths_list = []  # Stores original path for each entry in above lists
    doc_metadata_for_tfidf_fitting = []  # Stores (path, qa_name) for documents included in TF-IDF

    print("\n--- Phase 0: Ingesting and Preprocessing Documents (with Caching) ---")
    for qa_name, pdf_paths in qa_to_pdf_paths_map.items():
        for pdf_path_str in pdf_paths:
            pdf_path = Path(pdf_path_str)
            cache_file_key = f"{pdf_path.stem}_{get_file_hash(pdf_path_str)}"  # More robust key
            cache_file_path = Path(CACHE_DIR) / f"{cache_file_key}.dat"  # .dat for shelve

            processed_tokens = None
            stem_to_raw_map = None
            use_cache = False

            if not FORCE_REPROCESS_ALL and cache_file_path.exists():
                try:
                    with shelve.open(str(cache_file_path), flag='r') as db:
                        if db.get("preprocessing_version") == PREPROCESSING_VERSION and \
                                db.get("file_hash") == get_file_hash(pdf_path_str):  # Double check hash
                            print(f"Using cached data for: {pdf_path.name}")
                            processed_tokens = db["processed_tokens"]
                            stem_to_raw_map = db["stem_to_raw_map"]
                            use_cache = True
                        else:
                            print(f"Cache stale (version or hash mismatch) for: {pdf_path.name}")
                except Exception as e:  # pylint: disable=broad-except
                    print(f"Error reading cache for {pdf_path.name}: {e}. Reprocessing.")

            if not use_cache:
                print(f"Processing: {pdf_path.name} (for QA: {qa_name})")
                raw_text = extract_text_from_pdf(pdf_path_str)
                if raw_text:
                    current_file_hash = get_file_hash(pdf_path_str)
                    processed_tokens, stem_to_raw_map = preprocess_text_and_map(raw_text, pdf_path_str)
                    try:
                        with shelve.open(str(cache_file_path), flag='c') as db:  # 'c' to create if not exists
                            db["processed_tokens"] = processed_tokens
                            db["stem_to_raw_map"] = stem_to_raw_map
                            db["file_hash"] = current_file_hash
                            db["original_path"] = pdf_path_str
                            db["preprocessing_version"] = PREPROCESSING_VERSION
                        print(f"Cached processed data for: {pdf_path.name}")
                    except Exception as e:  # pylint: disable=broad-except
                        print(f"Error writing cache for {pdf_path.name}: {e}")
                else:
                    processed_tokens = []
                    stem_to_raw_map = defaultdict(set)

            if processed_tokens is not None:  # Ensure it was either cached or processed
                all_docs_processed_tokens_list.append(processed_tokens)
                all_docs_stem_to_raw_maps_list.append(stem_to_raw_map)
                all_docs_original_paths_list.append(pdf_path_str)
                doc_metadata_for_tfidf_fitting.append(
                    {'path': pdf_path_str, 'qa': qa_name, 'global_idx': len(all_docs_original_paths_list) - 1})

    if not all_docs_processed_tokens_list:
        print("No documents were successfully processed or loaded from cache. Exiting.")
        return

    all_processed_docs_joined_texts = [" ".join(tokens) for tokens in all_docs_processed_tokens_list]

    # --- Phase 2: TF-IDF Vectorizer Fitting (Global Scope) ---
    # We can also cache the fitted vectorizer if the corpus is large and stable
    fitted_vectorizer_cache_path = Path(CACHE_DIR) / "fitted_tfidf_vectorizer.shelf"
    corpus_vocabulary_hash = hashlib.md5(" ".join(
        sorted(list(set(token for doc_tokens in all_docs_processed_tokens_list for token in doc_tokens)))).encode(
        'utf-8')).hexdigest()

    fitted_vectorizer = None
    if not FORCE_REFIT_TFIDF and fitted_vectorizer_cache_path.exists():
        try:
            with shelve.open(str(fitted_vectorizer_cache_path), flag='r') as db:
                if db.get("preprocessing_version") == PREPROCESSING_VERSION and \
                        db.get("corpus_vocabulary_hash") == corpus_vocabulary_hash:
                    print("Loading cached TF-IDF vectorizer.")
                    fitted_vectorizer = db["vectorizer"]
        except Exception as e:  # pylint: disable=broad-except
            print(f"Could not load cached TF-IDF vectorizer: {e}. Will re-fit.")

    if fitted_vectorizer is None:
        # Pass None to calculate_corpus_tfidf so it fits a new one
        _, temp_fitted_vectorizer = calculate_corpus_tfidf(all_processed_docs_joined_texts,
                                                           all_docs_stem_to_raw_maps_list, [], fitted_vectorizer=None)
        if temp_fitted_vectorizer:
            fitted_vectorizer = temp_fitted_vectorizer
            try:
                with shelve.open(str(fitted_vectorizer_cache_path), flag='c') as db:
                    db["vectorizer"] = fitted_vectorizer
                    db["preprocessing_version"] = PREPROCESSING_VERSION
                    db["corpus_vocabulary_hash"] = corpus_vocabulary_hash
                print("Cached newly fitted TF-IDF vectorizer.")
            except Exception as e:  # pylint: disable=broad-except
                print(f"Error caching fitted TF-IDF vectorizer: {e}")
        else:
            print("Critical error: TF-IDF vectorizer could not be fitted. Cannot proceed with TF-IDF analysis.")
            # Allow N-gram/Collocation to proceed if desired, or exit
            # return

    # --- Phase 3: Analysis and Output (Per QA) ---
    for qa_name_key, pdf_paths_for_qa in qa_to_pdf_paths_map.items():
        # qa_name_key is the original casing from SEED_WORDS_RAW_MAP
        print(f"\n--- Processing Quality Attribute: {qa_name_key} ---")

        raw_seed_words_for_qa = SEED_WORDS_RAW_MAP.get(qa_name_key, [])
        if not raw_seed_words_for_qa:
            print(f"Warning: No seed words retrieved from SEED_WORDS_RAW_MAP for QA '{qa_name_key}'. Check mapping.")

        processed_seeds_for_qa = process_seed_keywords(raw_seed_words_for_qa, expand_synonyms=EXPAND_SEED_SYNONYMS)
        # print(f"Seeds for {qa_name_key}: {len(processed_seeds_for_qa)} processed seed terms.")

        # Identify indices of documents belonging to the current QA within the global lists
        qa_doc_global_indices = [
            meta['global_idx'] for meta in doc_metadata_for_tfidf_fitting if meta['qa'] == qa_name_key
        ]

        if not qa_doc_global_indices:
            print(f"No successfully processed documents found for QA: {qa_name_key}. Skipping analysis for this QA.")
            continue

        # TF-IDF for current QA's documents using the globally (potentially cached) fitted vectorizer
        qa_tfidf_dfs_list, _ = calculate_corpus_tfidf(  # We don't need to get the vectorizer back here
            all_processed_docs_joined_texts,
            all_docs_stem_to_raw_maps_list,
            qa_doc_global_indices,  # Pass only indices for this QA
            fitted_vectorizer=fitted_vectorizer  # Pass the globally fitted one
        )

        # Aggregate tokens and maps for this QA for N-gram/Collocation
        qa_combined_processed_tokens = []
        qa_combined_stem_to_raw_map = defaultdict(set)
        for global_idx in qa_doc_global_indices:
            qa_combined_processed_tokens.extend(all_docs_processed_tokens_list[global_idx])
            for stem, raw_set in all_docs_stem_to_raw_maps_list[global_idx].items():
                qa_combined_stem_to_raw_map[stem].update(raw_set)

        # print(f"Total processed tokens for {qa_name_key} (combined for Ngram/Colloc): {len(qa_combined_processed_tokens)}")

        # N-gram and Collocation Analysis for the current QA
        df_bigrams_qa = extract_ngrams(qa_combined_processed_tokens, qa_combined_stem_to_raw_map, n=2,
                                       num_top_ngrams=NUM_TOP_BIGRAMS)
        df_trigrams_qa = extract_ngrams(qa_combined_processed_tokens, qa_combined_stem_to_raw_map, n=3,
                                        num_top_ngrams=NUM_TOP_TRIGRAMS)
        df_colloc_general_qa = find_collocations_general(
            qa_combined_processed_tokens, qa_combined_stem_to_raw_map,
            num_collocations=NUM_COLLOCATIONS_GENERAL, window_size=COLLOCATION_WINDOW_SIZE
        )
        df_colloc_seeds_qa = find_collocations_with_seeds(
            qa_combined_processed_tokens, qa_combined_stem_to_raw_map, processed_seeds_for_qa,
            num_collocations=NUM_COLLOCATIONS_SEEDS, window_size=COLLOCATION_WINDOW_SIZE
        )

        # Save results for the current QA
        safe_qa_name = "".join(c if c.isalnum() else "_" for c in qa_name_key)  # Use qa_name_key
        output_excel_qa_path = f"keyword_analysis_{safe_qa_name}.xlsx"
        print(f"--- Saving results for {qa_name_key} to {output_excel_qa_path} ---")
        try:
            with pd.ExcelWriter(output_excel_qa_path, engine='openpyxl') as writer:
                for i, df_doc_tfidf in enumerate(qa_tfidf_dfs_list):
                    if not df_doc_tfidf.empty:
                        original_pdf_path_for_sheet = all_docs_original_paths_list[qa_doc_global_indices[i]]
                        sheet_name_pdf = Path(original_pdf_path_for_sheet).stem[:25]
                        unique_sheet_name_pdf = sheet_name_pdf
                        _count = 1
                        while f"TFIDF_{unique_sheet_name_pdf}" in writer.sheets:
                            unique_sheet_name_pdf = f"{sheet_name_pdf}_{_count}"
                            _count += 1
                            if len(unique_sheet_name_pdf) > 25:  # keep it short
                                unique_sheet_name_pdf = f"Doc{qa_doc_global_indices[i]}"
                                break
                        df_doc_tfidf.to_excel(writer, sheet_name=f"TFIDF_{unique_sheet_name_pdf}", index=False)

                if not df_bigrams_qa.empty: df_bigrams_qa.to_excel(writer, sheet_name='QA_Top_Bigrams', index=False)
                if not df_trigrams_qa.empty: df_trigrams_qa.to_excel(writer, sheet_name='QA_Top_Trigrams', index=False)
                if not df_colloc_general_qa.empty: df_colloc_general_qa.to_excel(writer, sheet_name='QA_Colloc_General',
                                                                                 index=False)
                if not df_colloc_seeds_qa.empty: df_colloc_seeds_qa.to_excel(writer, sheet_name='QA_Colloc_Seeds',
                                                                             index=False)
            print(f"Successfully saved Excel for {qa_name_key}.")
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error saving Excel for {qa_name_key}: {e}")

    print("\nCorpus analysis finished.")


if __name__ == "__main__":
    main()