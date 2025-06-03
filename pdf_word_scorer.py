import os
import PyPDF2
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
from nltk import WordNetLemmatizer, ngrams as nltk_ngrams
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.probability import FreqDist

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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
def calculate_qa_level_tfidf_aggregates(
        list_of_doc_analysis_dfs,  # Now expects DFs with TF, IDF columns
        qa_combined_stem_to_raw_map,
        global_term_idf_map,  # Pass this in
        global_term_doc_count_map  # Pass this in
):
    if not list_of_doc_analysis_dfs:
        return pd.DataFrame()

    term_sum_tfidf = defaultdict(float)
    term_sum_tf = defaultdict(float)  # For Summed TF in QA
    term_doc_count_in_qa = defaultdict(int)
    all_terms_in_qa_docs = set()

    for df_doc in list_of_doc_analysis_dfs:
        if df_doc.empty or not all(
                col in df_doc.columns for col in ['Term (Processed)', 'TF-IDF Score', 'TF (in doc)']):
            continue

        unique_terms_in_this_doc = set()
        for _, row in df_doc.iterrows():
            term = row['Term (Processed)']
            tfidf_score = row['TF-IDF Score']
            tf_score = row['TF (in doc)']

            term_sum_tfidf[term] += tfidf_score
            term_sum_tf[term] += tf_score  # Summing raw TF
            all_terms_in_qa_docs.add(term)
            unique_terms_in_this_doc.add(term)

        for term in unique_terms_in_this_doc:
            term_doc_count_in_qa[term] += 1

    if not all_terms_in_qa_docs:
        return pd.DataFrame()

    aggregated_data = []
    for term in sorted(list(all_terms_in_qa_docs)):
        sum_tfidf = term_sum_tfidf[term]
        sum_tf_qa = term_sum_tf[term]  # Summed TF for this term within QA
        doc_count_qa = term_doc_count_in_qa[term]

        avg_tfidf_qa = sum_tfidf / doc_count_qa if doc_count_qa > 0 else 0.0
        avg_tf_qa = sum_tf_qa / doc_count_qa if doc_count_qa > 0 else 0.0  # Average TF in QA docs with term

        idf_global = global_term_idf_map.get(term, 0.0)  # Get from passed map
        global_doc_freq = global_term_doc_count_map.get(term, 0)  # Get from passed map

        original_words = format_original_words(qa_combined_stem_to_raw_map.get(term, {term}))

        aggregated_data.append({
            'Term (Processed)': term,
            'Original Word(s)': original_words,
            'Summed TF-IDF (QA)': sum_tfidf,
            'Avg TF-IDF (QA docs with term)': avg_tfidf_qa,
            'Summed TF (QA)': sum_tf_qa,
            'Avg TF (QA docs with term)': avg_tf_qa,
            'IDF (global)': idf_global,
            'Global Doc Freq': global_doc_freq,
            'Doc Count (in QA)': doc_count_qa
        })

    df_aggregated = pd.DataFrame(aggregated_data)
    df_aggregated = df_aggregated.sort_values(
        by=['Summed TF-IDF (QA)', 'Avg TF-IDF (QA docs with term)'],
        ascending=[False, False]
    ).reset_index(drop=True)

    return df_aggregated

# --- Modified Helper Function for Preparing Seed Keywords Sheet Data (from previous step) ---
def prepare_seed_keywords_sheet_data(raw_seed_keywords_list):
    sheet_data = []
    if not raw_seed_keywords_list:
        return sheet_data
    for raw_seed in raw_seed_keywords_list:
        processed_tokens, _ = preprocess_text_and_map(raw_seed.lower())
        processed_seed_display = " ".join(processed_tokens) if processed_tokens else raw_seed.lower() # Fallback
        sheet_data.append({
            'Raw Seed Keyword': raw_seed,
            'Processed Seed Keyword': processed_seed_display
        })
    return sheet_data

def extract_text_from_document(doc_path):
    """
    Extracts text from a PDF or TXT file.
    """
    doc_path_lower = doc_path.lower()
    if doc_path_lower.endswith(".pdf"):
        return extract_text_from_pdf(doc_path) # Reuse existing PDF extraction
    elif doc_path_lower.endswith(".txt"):
        try:
            with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f: # Added errors='ignore' for robustness
                text = f.read()
            # print(f"Successfully extracted text (length: {len(text)} characters) from {Path(doc_path).name}.")
            return text
        except FileNotFoundError:
            print(f"Error: TXT file not found at {doc_path}")
            return None
        except Exception as e:
            print(f"Error reading TXT file {doc_path}: {e}")
            return None
    else:
        print(f"Unsupported file type: {doc_path}. Skipping.")
        return None

def find_documents_by_qa(base_dir):
    """
    Walks through the base_dir, finds PDFs and TXTs in immediate subdirectories (QAs).
    Returns a dictionary: {qa_name: [list_of_document_paths]}
    """
    qa_to_documents = defaultdict(list)  # Changed variable name
    if not os.path.isdir(base_dir):
        print(f"Error: Base document directory '{base_dir}' not found.")
        return qa_to_documents

    for dir_item_name in os.listdir(base_dir):
        item_path = os.path.join(base_dir, dir_item_name)
        if os.path.isdir(item_path):
            matched_qa_key = next(
                (key_in_map for key_in_map in SEED_WORDS_RAW_MAP.keys() if key_in_map.lower() == dir_item_name.lower()),
                None)

            if matched_qa_key:
                print(f"Found QA directory: {dir_item_name} (mapped to key: '{matched_qa_key}')")
                for root, _, files in os.walk(item_path):
                    for file in files:
                        file_lower = file.lower()
                        if file_lower.endswith(".pdf") or file_lower.endswith(".txt"):  # Added .txt check
                            doc_path = os.path.join(root, file)
                            qa_to_documents[matched_qa_key].append(doc_path)
            else:
                print(f"Skipping directory '{dir_item_name}' as it's not a recognized QA in SEED_WORDS_RAW_MAP.")
    return qa_to_documents

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

def get_term_frequencies(processed_doc_text_list):
    """
    Calculates raw term frequencies for a list of documents.
    Args:
        processed_doc_text_list: List of strings, each a space-joined doc.
    Returns:
        A sparse matrix of TF counts (docs x terms), and the CountVectorizer.
    """
    tf_vectorizer = CountVectorizer(ngram_range=(1,1)) # Ensure same tokenization/params as Tfidf
    tf_matrix = tf_vectorizer.fit_transform(processed_doc_text_list)
    return tf_matrix, tf_vectorizer


def calculate_corpus_tfidf_with_components(
        all_processed_docs_texts,
        all_stem_to_raw_maps_list,
        target_doc_indices,
        fitted_tfidf_vectorizer=None,
        fitted_tf_vectorizer=None,  # For global TF and doc counts
        global_term_idf_map=None,  # Term -> IDF score
        global_term_doc_count_map=None  # Term -> global document count
):
    """
    Calculates TF-IDF, TF, and IDF for target documents.
    Manages fitting of vectorizers if not provided.
    """
    num_total_corpus_docs = len(all_processed_docs_texts)

    if not all_processed_docs_texts:
        return [], None, None, None, None

    # 1. Fit/use CountVectorizer for TF and Global Document Frequencies
    if fitted_tf_vectorizer is None:
        print(f"Fitting CountVectorizer on {num_total_corpus_docs} documents for TF and global doc counts...")
        tf_matrix_full_corpus, temp_tf_vectorizer = get_term_frequencies(all_processed_docs_texts)
        if temp_tf_vectorizer:
            fitted_tf_vectorizer = temp_tf_vectorizer
            # Build global_term_doc_count_map from the full corpus TF matrix
            temp_global_term_doc_count_map = defaultdict(int)
            tf_vocab = fitted_tf_vectorizer.vocabulary_  # term -> index
            # Sum occurrences across documents (df_t)
            doc_counts_per_term = tf_matrix_full_corpus.sum(axis=0).A1  # A1 flattens
            for term, index in tf_vocab.items():
                temp_global_term_doc_count_map[term] = int(doc_counts_per_term[index] > 0) * tf_matrix_full_corpus[:,
                                                                                             index].count_nonzero()

            global_term_doc_count_map = temp_global_term_doc_count_map
        else:
            print("Error: CountVectorizer could not be fitted.")
            # Return structure matching successful run but with empty DFs
            return [pd.DataFrame() for _ in target_doc_indices], None, None, None, None
    else:
        print("Using pre-fitted CountVectorizer...")
        tf_matrix_full_corpus = fitted_tf_vectorizer.transform(
            all_processed_docs_texts)  # Just transform for consistency if needed later

    # 2. Fit/use TfidfVectorizer for TF-IDF and IDF scores
    if fitted_tfidf_vectorizer is None:
        # Use vocabulary from CountVectorizer to ensure consistency if different tokenization happens
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=2, max_df=.5, vocabulary=fitted_tf_vectorizer.vocabulary_)
        print(f"Fitting TF-IDF vectorizer on {num_total_corpus_docs} documents...")
        try:
            corpus_tfidf_matrix = tfidf_vectorizer.fit_transform(all_processed_docs_texts)
            fitted_tfidf_vectorizer = tfidf_vectorizer
            # Build global_term_idf_map
            temp_global_term_idf_map = defaultdict(float)
            idf_vocab = fitted_tfidf_vectorizer.vocabulary_  # term -> index
            idf_scores = fitted_tfidf_vectorizer.idf_  # index -> idf
            for term, index in idf_vocab.items():
                temp_global_term_idf_map[term] = idf_scores[index]
            global_term_idf_map = temp_global_term_idf_map
        except ValueError as e:
            print(f"Could not fit TF-IDF vectorizer. Reason: {e}")
            return [pd.DataFrame() for _ in
                    target_doc_indices], fitted_tf_vectorizer, None, global_term_doc_count_map, None
    else:
        print("Using pre-fitted TF-IDF vectorizer...")
        tfidf_vectorizer = fitted_tfidf_vectorizer
        try:
            corpus_tfidf_matrix = tfidf_vectorizer.transform(all_processed_docs_texts)
        except Exception as e:
            print(f"Error transforming with pre-fitted TF-IDF vectorizer: {e}")
            return [pd.DataFrame() for _ in
                    target_doc_indices], fitted_tf_vectorizer, fitted_tfidf_vectorizer, global_term_doc_count_map, global_term_idf_map

    tfidf_feature_names = fitted_tfidf_vectorizer.get_feature_names_out()
    tf_feature_names = fitted_tf_vectorizer.get_feature_names_out()  # Should be same if vocab was passed

    results_dfs = []

    for doc_idx in target_doc_indices:
        if doc_idx >= num_total_corpus_docs:  # Should use num_total_corpus_docs which is len(all_processed_docs_texts)
            print(f"Warning: Document index {doc_idx} out of bounds. Skipping.")
            results_dfs.append(pd.DataFrame())
            continue

        doc_tfidf_vector = corpus_tfidf_matrix[doc_idx]
        doc_tf_vector = tf_matrix_full_corpus[doc_idx]  # Get TF vector for this doc

        doc_data = []
        current_doc_stem_to_raw_map = all_stem_to_raw_maps_list[doc_idx]

        # Iterate through terms present in the document based on TF-IDF vector
        for col_idx in doc_tfidf_vector.nonzero()[1]:
            term = tfidf_feature_names[col_idx]
            tfidf_score = doc_tfidf_vector[0, col_idx]

            # Get TF score. tf_feature_names should align if vocab was shared
            # Need to find the index of 'term' in tf_vectorizer's vocabulary
            tf_term_idx = fitted_tf_vectorizer.vocabulary_.get(term)
            tf_score = 0
            if tf_term_idx is not None and tf_term_idx < doc_tf_vector.shape[1]:
                tf_score = doc_tf_vector[0, tf_term_idx]

            idf_score = global_term_idf_map.get(term, 0.0)  # Get from global map
            global_doc_freq = global_term_doc_count_map.get(term, 0)  # Get from global map

            original_words = format_original_words(current_doc_stem_to_raw_map.get(term, {term}))

            doc_data.append({
                'Term (Processed)': term,
                'Original Word(s)': original_words,
                'TF (in doc)': tf_score,
                'IDF (global)': idf_score,
                'TF-IDF Score': tfidf_score,
                'Global Doc Freq': global_doc_freq
            })

        if not doc_data:
            results_dfs.append(pd.DataFrame())
            continue

        df_doc_analysis = pd.DataFrame(doc_data)
        df_doc_analysis = df_doc_analysis.sort_values(by='TF-IDF Score', ascending=False).reset_index(drop=True)
        results_dfs.append(df_doc_analysis)

    return results_dfs, fitted_tf_vectorizer, fitted_tfidf_vectorizer, global_term_doc_count_map, global_term_idf_map


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
    qa_to_doc_paths_map  = find_documents_by_qa(BASE_PDF_DIR)

    if not qa_to_doc_paths_map :
        print("No PDFs found or QA directories configured. Exiting.")
        return

    # --- Phase 0: Cache Management and Corpus Ingestion ---
    all_docs_processed_tokens_list = []
    all_docs_stem_to_raw_maps_list = []
    all_docs_original_paths_list = []  # Stores original path for each entry in above lists
    doc_metadata_for_tfidf_fitting = []  # Stores (path, qa_name) for documents included in TF-IDF

    print("\n--- Phase 0: Ingesting and Preprocessing Documents (with Caching) ---")
    for qa_name, doc_paths in qa_to_doc_paths_map .items():
        for pdf_path_str in doc_paths:
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
                raw_text = extract_text_from_document(pdf_path_str)
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

    fitted_tf_vectorizer_cache_path = Path(CACHE_DIR) / "fitted_tf_vectorizer.shelf"
    fitted_tfidf_vectorizer_cache_path = Path(CACHE_DIR) / "fitted_tfidf_vectorizer.shelf"
    global_maps_cache_path = Path(CACHE_DIR) / "global_term_maps.shelf"  # For IDF and DocCount maps

    # Hash based on joined texts for TF vectorizer, as its vocab depends on it
    corpus_raw_text_hash = hashlib.md5("".join(all_processed_docs_joined_texts).encode('utf-8')).hexdigest()

    fitted_tf_vectorizer = None
    fitted_tfidf_vectorizer = None
    global_term_idf_map = None
    global_term_doc_count_map = None

    # Try loading all from cache first
    # For TF vectorizer and global doc count map
    if not FORCE_REFIT_TFIDF and fitted_tf_vectorizer_cache_path.exists() and global_maps_cache_path.exists():
        try:
            with shelve.open(str(fitted_tf_vectorizer_cache_path), flag='r') as db_tf, \
                    shelve.open(str(global_maps_cache_path), flag='r') as db_maps:
                if db_tf.get("preprocessing_version") == PREPROCESSING_VERSION and \
                        db_tf.get("corpus_raw_text_hash") == corpus_raw_text_hash and \
                        db_maps.get("preprocessing_version") == PREPROCESSING_VERSION and \
                        db_maps.get("corpus_raw_text_hash") == corpus_raw_text_hash:
                    print("Loading cached TF vectorizer and global term maps.")
                    fitted_tf_vectorizer = db_tf["tf_vectorizer"]
                    global_term_doc_count_map = db_maps["global_term_doc_count_map"]
        except Exception as e:
            print(f"Could not load cached TF vectorizer or global maps: {e}. Will re-fit.")

    # For TF-IDF vectorizer and global IDF map (depends on TF vectorizer's vocab)
    if fitted_tf_vectorizer and not FORCE_REFIT_TFIDF and fitted_tfidf_vectorizer_cache_path.exists() and global_maps_cache_path.exists():  # Check if TF vectorizer was loaded
        try:
            with shelve.open(str(fitted_tfidf_vectorizer_cache_path), flag='r') as db_tfidf, \
                    shelve.open(str(global_maps_cache_path), flag='r') as db_maps:  # db_maps for idf map
                # Check if vocab hash from TF vectorizer matches what TFIDF was trained on
                # This is tricky if vocabulary_ arg was used. For simplicity, rely on PP_VERSION and raw_text_hash
                if db_tfidf.get("preprocessing_version") == PREPROCESSING_VERSION and \
                        db_tfidf.get("corpus_raw_text_hash") == corpus_raw_text_hash and \
                        db_maps.get("preprocessing_version") == PREPROCESSING_VERSION and \
                        db_maps.get("corpus_raw_text_hash") == corpus_raw_text_hash:
                    print("Loading cached TF-IDF vectorizer.")
                    fitted_tfidf_vectorizer = db_tfidf["tfidf_vectorizer"]
                    global_term_idf_map = db_maps["global_term_idf_map"]  # Load idf map
        except Exception as e:
            print(f"Could not load cached TF-IDF vectorizer: {e}. Will re-fit if needed.")

    # If any component is missing, refit them in sequence
    if not all([fitted_tf_vectorizer, fitted_tfidf_vectorizer, global_term_doc_count_map, global_term_idf_map]):
        print("One or more vectorizer/map components missing from cache or outdated, re-calculating...")
        # Call the function to fit/get all components
        _, \
        temp_fitted_tf_vectorizer, \
        temp_fitted_tfidf_vectorizer, \
        temp_global_term_doc_count_map, \
        temp_global_term_idf_map = calculate_corpus_tfidf_with_components(
            all_processed_docs_joined_texts,
            all_docs_stem_to_raw_maps_list,  # Needed for context but not directly by this call if just fitting
            [],  # No target docs needed for fitting phase
            fitted_tfidf_vectorizer=None,  # Force refit if here
            fitted_tf_vectorizer=None,  # Force refit if here
            global_term_idf_map=None,
            global_term_doc_count_map=None
        )
        # Assign successfully fitted components
        if temp_fitted_tf_vectorizer: fitted_tf_vectorizer = temp_fitted_tf_vectorizer
        if temp_fitted_tfidf_vectorizer: fitted_tfidf_vectorizer = temp_fitted_tfidf_vectorizer
        if temp_global_term_doc_count_map: global_term_doc_count_map = temp_global_term_doc_count_map
        if temp_global_term_idf_map: global_term_idf_map = temp_global_term_idf_map

        # Cache the newly fitted components
        if fitted_tf_vectorizer:
            try:
                with shelve.open(str(fitted_tf_vectorizer_cache_path), flag='c') as db:
                    db["tf_vectorizer"] = fitted_tf_vectorizer
                    db["preprocessing_version"] = PREPROCESSING_VERSION
                    db["corpus_raw_text_hash"] = corpus_raw_text_hash
                print("Cached TF vectorizer.")
            except Exception as e:
                print(f"Error caching TF vectorizer: {e}")

        if fitted_tfidf_vectorizer:  # Only cache if TF-IDF fitting was also successful
            try:
                with shelve.open(str(fitted_tfidf_vectorizer_cache_path), flag='c') as db:
                    db["tfidf_vectorizer"] = fitted_tfidf_vectorizer
                    db["preprocessing_version"] = PREPROCESSING_VERSION
                    db["corpus_raw_text_hash"] = corpus_raw_text_hash  # Assuming TFIDF vocab depends on this
                print("Cached TF-IDF vectorizer.")
            except Exception as e:
                print(f"Error caching TF-IDF vectorizer: {e}")

        if global_term_doc_count_map is not None and global_term_idf_map is not None:  # Check for None explicitly
            try:
                with shelve.open(str(global_maps_cache_path), flag='c') as db:
                    db["global_term_doc_count_map"] = global_term_doc_count_map
                    db["global_term_idf_map"] = global_term_idf_map
                    db["preprocessing_version"] = PREPROCESSING_VERSION
                    db["corpus_raw_text_hash"] = corpus_raw_text_hash
                print("Cached global term maps (IDF and DocCount).")
            except Exception as e:
                print(f"Error caching global term maps: {e}")

    if not all([fitted_tf_vectorizer, fitted_tfidf_vectorizer, global_term_doc_count_map, global_term_idf_map]):
        print(
            "Critical error: Vectorizers or global term maps could not be initialized. Exiting TF-IDF related analysis.")
        # Decide if you want to proceed with N-gram/Collocation or exit entirely
        # For now, we'll let it try, but TF-IDF sheets will be empty.
        # return

    # --- Phase 3: Analysis and Output (Per QA) ---
    for qa_name_key, pdf_paths_for_qa in qa_to_doc_paths_map .items():
        print(f"\n--- Processing Quality Attribute: {qa_name_key} ---")

        raw_seed_words_for_qa = SEED_WORDS_RAW_MAP.get(qa_name_key, [])
        processed_seeds_for_qa = process_seed_keywords(raw_seed_words_for_qa, expand_synonyms=EXPAND_SEED_SYNONYMS)

        qa_doc_global_indices = [
            meta['global_idx'] for meta in doc_metadata_for_tfidf_fitting if meta['qa'] == qa_name_key
        ]

        if not qa_doc_global_indices:
            # ... (skip QA if no docs - same) ...
            continue

        # Get per-document TF-IDF, TF, IDF details using the globally fitted vectorizers/maps
        qa_doc_analysis_dfs, _, _, _, _ = calculate_corpus_tfidf_with_components(
            all_processed_docs_joined_texts,
            all_docs_stem_to_raw_maps_list,
            qa_doc_global_indices,
            fitted_tfidf_vectorizer=fitted_tfidf_vectorizer,  # Pass fitted
            fitted_tf_vectorizer=fitted_tf_vectorizer,  # Pass fitted
            global_term_idf_map=global_term_idf_map,
            global_term_doc_count_map=global_term_doc_count_map
        )

        # Aggregate tokens and maps for this QA
        qa_combined_processed_tokens = []
        qa_combined_stem_to_raw_map = defaultdict(set)
        for global_idx in qa_doc_global_indices:
            qa_combined_processed_tokens.extend(all_docs_processed_tokens_list[global_idx])
            for stem, raw_set in all_docs_stem_to_raw_maps_list[global_idx].items():
                qa_combined_stem_to_raw_map[stem].update(raw_set)

        # Calculate QA-level TF-IDF aggregates (now includes TF, IDF columns)
        df_qa_tfidf_aggregates = calculate_qa_level_tfidf_aggregates(
            qa_doc_analysis_dfs,  # Pass the list of DFs with TF/IDF components
            qa_combined_stem_to_raw_map,
            global_term_idf_map if global_term_idf_map else defaultdict(float),  # Pass maps, provide default if None
            global_term_doc_count_map if global_term_doc_count_map else defaultdict(int)
        )

        # ... (N-gram, Collocation analysis - same) ...
        df_bigrams_qa = extract_ngrams(qa_combined_processed_tokens, qa_combined_stem_to_raw_map, n=2,
                                       num_top_ngrams=NUM_TOP_BIGRAMS)
        df_trigrams_qa = extract_ngrams(qa_combined_processed_tokens, qa_combined_stem_to_raw_map, n=3,
                                        num_top_ngrams=NUM_TOP_TRIGRAMS)
        df_colloc_general_qa = find_collocations_general(qa_combined_processed_tokens, qa_combined_stem_to_raw_map,
                                                         num_collocations=NUM_COLLOCATIONS_GENERAL,
                                                         window_size=COLLOCATION_WINDOW_SIZE)
        df_colloc_seeds_qa = find_collocations_with_seeds(qa_combined_processed_tokens, qa_combined_stem_to_raw_map,
                                                          processed_seeds_for_qa,
                                                          num_collocations=NUM_COLLOCATIONS_SEEDS,
                                                          window_size=COLLOCATION_WINDOW_SIZE)

        # Prepare seed keywords sheet data (same)
        seed_keywords_sheet_data = prepare_seed_keywords_sheet_data(raw_seed_words_for_qa)
        df_seed_keywords_qa = pd.DataFrame(seed_keywords_sheet_data) if seed_keywords_sheet_data else pd.DataFrame()

        # Save results to Excel
        safe_qa_name = "".join(c if c.isalnum() else "_" for c in qa_name_key)
        output_excel_qa_path = f"keyword_analysis_{safe_qa_name}.xlsx"
        print(f"--- Saving results for {qa_name_key} to {output_excel_qa_path} ---")
        try:
            with pd.ExcelWriter(output_excel_qa_path, engine='openpyxl') as writer:
                if not df_qa_tfidf_aggregates.empty:
                    df_qa_tfidf_aggregates.to_excel(writer, sheet_name='QA_TFIDF_Aggregates', index=False)

                if not df_seed_keywords_qa.empty:
                    df_seed_keywords_qa.to_excel(writer, sheet_name='QA_Seed_Keywords', index=False)

                # Save individual document TF-IDF (now with TF, IDF columns)
                for i, df_doc_analysis in enumerate(qa_doc_analysis_dfs):  # Iterate through the list of DFs
                    if not df_doc_analysis.empty:
                        # ... (sheet naming logic for individual analysis sheets - same) ...
                        original_pdf_path_for_sheet = all_docs_original_paths_list[qa_doc_global_indices[i]]
                        sheet_name_base = Path(original_pdf_path_for_sheet).stem
                        sheet_name_pdf = "".join(c if c.isalnum() else "_" for c in sheet_name_base)[
                                         :20]  # Shorter name
                        unique_sheet_name_pdf = sheet_name_pdf
                        _count = 1
                        while f"DocAn_{unique_sheet_name_pdf}" in writer.sheets:
                            unique_sheet_name_pdf = f"{sheet_name_pdf}_{_count}"
                            _count += 1
                            if len(unique_sheet_name_pdf) > 20:
                                unique_sheet_name_pdf = f"Doc{qa_doc_global_indices[i]}"
                                if f"DocAn_{unique_sheet_name_pdf}" in writer.sheets:
                                    unique_sheet_name_pdf = f"Doc{qa_doc_global_indices[i]}_{_count}"
                                break
                        df_doc_analysis.to_excel(writer, sheet_name=f"DocAn_{unique_sheet_name_pdf}",
                                                 index=False)  # Renamed sheet prefix

                # ... (Save other N-gram, Collocation sheets - same) ...
                if not df_bigrams_qa.empty: df_bigrams_qa.to_excel(writer, sheet_name='QA_Top_Bigrams', index=False)
                if not df_trigrams_qa.empty: df_trigrams_qa.to_excel(writer, sheet_name='QA_Top_Trigrams', index=False)
                if not df_colloc_general_qa.empty: df_colloc_general_qa.to_excel(writer, sheet_name='QA_Colloc_General',
                                                                                 index=False)
                if not df_colloc_seeds_qa.empty: df_colloc_seeds_qa.to_excel(writer, sheet_name='QA_Colloc_Seeds',
                                                                             index=False)

            print(f"Successfully saved Excel for {qa_name_key}.")
        except Exception as e:
            print(f"Error saving Excel for {qa_name_key}: {e}")

    print("\nCorpus analysis finished.")


if __name__ == "__main__":
    main()