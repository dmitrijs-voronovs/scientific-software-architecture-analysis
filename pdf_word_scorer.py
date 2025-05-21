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
from collections import defaultdict # For stem_to_raw_map

from quality_attributes import quality_attributes

# --- Configuration ---
# SEED_KEYWORDS_RAW = [
#     "performance", "latency", "throughput", "response", "load", "scalability", "bottleneck", # Corrected scalab
#     "availability", "downtime", "reliability", "fault", "failure", "resilience", "recovery", # Corrected reliab, resilien, recover
#     "security", "attack", "vulnerability", "encryption", "authorization", "authentication", # Corrected vulnerab, encrypt, authoriz, authentcat
# ]
SEED_WORDS_RAW_MAP  = quality_attributes


USE_STEMMING = True
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# --- Helper Functions ---

def find_pdfs_by_qa(base_dir):
    """
    Walks through the base_dir, finds PDFs in immediate subdirectories (QAs).
    Returns a dictionary: {qa_name: [list_of_pdf_paths]}
    """
    qa_to_pdfs = defaultdict(list)
    if not os.path.isdir(base_dir):
        print(f"Error: Base PDF directory '{base_dir}' not found.")
        return qa_to_pdfs

    for qa_name in os.listdir(base_dir):
        qa_path = os.path.join(base_dir, qa_name)
        if os.path.isdir(qa_path):
            # Check if this directory name is a key in our SEED_WORDS_RAW_MAP
            # This ensures we only process directories we have seeds for (optional, but good for consistency)
            if qa_name.lower() in (key.lower() for key in SEED_WORDS_RAW_MAP.keys()): # Case-insensitive match for dir name
                print(f"Found QA directory: {qa_name}")
                for root, _, files in os.walk(qa_path): # os.walk for nested PDFs within a QA dir
                    for file in files:
                        if file.lower().endswith(".pdf"):
                            pdf_path = os.path.join(root, file)
                            qa_to_pdfs[qa_name].append(pdf_path) # Use original qa_name from listdir
                            # print(f"  - Found PDF: {pdf_path}")
            else:
                print(f"Skipping directory '{qa_name}' as it's not in SEED_WORDS_RAW_MAP or doesn't match QA naming.")
    return qa_to_pdfs


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

def calculate_corpus_tfidf(all_processed_docs_texts, all_stem_to_raw_maps, target_doc_indices):
    """
    Calculates TF-IDF scores for target documents using a corpus-wide IDF.
    Args:
        all_processed_docs_texts: List of strings, where each string is a join of processed tokens for a doc.
        all_stem_to_raw_maps: List of stem_to_raw_map dictionaries, one for each document in all_processed_docs_texts.
                              The order must correspond to all_processed_docs_texts.
        target_doc_indices: List of indices indicating which documents in all_processed_docs_texts
                            are the ones we want to get TF-IDF scores for (e.g., docs for current QA).
    Returns:
        A list of DataFrames, one for each target document.
    """
    if not all_processed_docs_texts:
        return []

    vectorizer = TfidfVectorizer(ngram_range=(1, 1)) # Can also do (1,2) for bigrams in TF-IDF

    print(f"Fitting TF-IDF vectorizer on {len(all_processed_docs_texts)} documents from the entire corpus...")
    try:
        # Fit on ALL documents in the corpus to get global IDF values
        corpus_tfidf_matrix = vectorizer.fit_transform(all_processed_docs_texts)
        feature_names = vectorizer.get_feature_names_out() # These are the processed terms from the whole corpus
    except ValueError as e:
        print(f"Could not fit TF-IDF vectorizer. Reason: {e}")
        return [pd.DataFrame(columns=['Term (Processed)', 'Original Word(s)', 'TF-IDF Score']) for _ in target_doc_indices]

    results_dfs = []
    for i, doc_idx in enumerate(target_doc_indices):
        if doc_idx >= len(all_processed_docs_texts) or doc_idx >= corpus_tfidf_matrix.shape[0]:
            print(f"Warning: Document index {doc_idx} out of bounds. Skipping TF-IDF for this doc.")
            results_dfs.append(pd.DataFrame(columns=['Term (Processed)', 'Original Word(s)', 'TF-IDF Score']))
            continue

        # Get the TF-IDF scores for the current target document
        # `corpus_tfidf_matrix` is a sparse matrix. `doc_idx` row contains scores for that doc.
        doc_vector = corpus_tfidf_matrix[doc_idx]
        # Create a mapping of feature index to score for this document
        term_scores = {feature_names[col_idx]: doc_vector[0, col_idx]
                       for col_idx in doc_vector.nonzero()[1]} # nonzero gets indices of terms present in doc

        if not term_scores:
            results_dfs.append(pd.DataFrame(columns=['Term (Processed)', 'Original Word(s)', 'TF-IDF Score']))
            continue

        current_doc_stem_to_raw_map = all_stem_to_raw_maps[doc_idx]

        df_tfidf = pd.DataFrame(list(term_scores.items()), columns=['Term (Processed)', 'TF-IDF Score'])
        df_tfidf['Original Word(s)'] = df_tfidf['Term (Processed)'].apply(
            lambda term: format_original_words(current_doc_stem_to_raw_map.get(term, set()))
        )
        df_tfidf = df_tfidf.sort_values(by='TF-IDF Score', ascending=False).reset_index(drop=True)
        df_tfidf = df_tfidf[['Term (Processed)', 'Original Word(s)', 'TF-IDF Score']] # Reorder

        results_dfs.append(df_tfidf)
        # print(f"  Calculated TF-IDF for document {i+1} of current QA (Original Index: {doc_idx}), {len(df_tfidf)} terms.")
    return results_dfs

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
    # --- Configuration from global scope ---
    BASE_PDF_DIR = "metadata/papers/"
    NUM_TOP_BIGRAMS = 100
    NUM_TOP_TRIGRAMS = 50
    NUM_COLLOCATIONS_GENERAL = 100
    NUM_COLLOCATIONS_SEEDS = 100
    COLLOCATION_WINDOW_SIZE = 5
    EXPAND_SEED_SYNONYMS = False  # Set to True to expand seed keywords with synonyms

    print("Starting CORPUS keyword analysis...")
    qa_to_pdf_paths = find_pdfs_by_qa(BASE_PDF_DIR)

    if not qa_to_pdf_paths:
        print("No PDFs found or QA directories configured. Exiting.")
        return

    # --- Step 1: Preprocess ALL documents in the corpus to build global IDF for TF-IDF ---
    all_docs_processed_tokens_list = []  # List of lists of tokens
    all_docs_stem_to_raw_maps_list = []  # List of stem_to_raw_map dicts
    all_docs_original_paths = []  # Keep track of original PDF paths

    print("\n--- Preprocessing all documents for global TF-IDF fitting ---")
    for qa_name, pdf_paths in qa_to_pdf_paths.items():
        for pdf_path in pdf_paths:
            print(f"Preprocessing for global IDF: {pdf_path}")
            raw_text = extract_text_from_pdf(pdf_path)
            if raw_text:
                processed_tokens, stem_to_raw_map = preprocess_text(raw_text)
                all_docs_processed_tokens_list.append(processed_tokens)
                all_docs_stem_to_raw_maps_list.append(stem_to_raw_map)
                all_docs_original_paths.append(pdf_path)
            else:  # Add empty entries to keep lists aligned if a PDF fails
                all_docs_processed_tokens_list.append([])
                all_docs_stem_to_raw_maps_list.append(defaultdict(set))
                all_docs_original_paths.append(pdf_path)  # Still add path for reference

    all_processed_docs_joined_texts = [" ".join(tokens) for tokens in all_docs_processed_tokens_list]

    # --- Step 2: Process each QA separately ---
    for qa_name_from_dir, pdf_paths_for_qa in qa_to_pdf_paths.items():
        print(f"\n--- Processing Quality Attribute: {qa_name_from_dir} ---")

        # Match directory name to SEED_WORDS_RAW_MAP key (case-insensitive)
        current_qa_seed_key = ""
        for key_in_map in SEED_WORDS_RAW_MAP.keys():
            if key_in_map.lower() == qa_name_from_dir.lower():
                current_qa_seed_key = key_in_map
                break

        if not current_qa_seed_key:
            print(
                f"Warning: No seed words found in SEED_WORDS_RAW_MAP for QA '{qa_name_from_dir}'. Skipping seed-based analysis for this QA.")
            raw_seed_words_for_qa = []
        else:
            raw_seed_words_for_qa = SEED_WORDS_RAW_MAP[current_qa_seed_key]

        processed_seeds_for_qa = process_seed_keywords(raw_seed_words_for_qa, expand_synonyms=EXPAND_SEED_SYNONYMS)
        print(f"Seeds for {qa_name_from_dir}: {len(processed_seeds_for_qa)} processed seed terms.")

        # Identify indices of documents belonging to the current QA within the global lists
        qa_doc_indices_global = [
            i for i, path in enumerate(all_docs_original_paths) if path in pdf_paths_for_qa
        ]

        if not qa_doc_indices_global:
            print(f"No successfully processed documents found for QA: {qa_name_from_dir}. Skipping analysis.")
            continue

        # --- TF-IDF calculation for current QA's documents using global IDF ---
        # This will return a list of DataFrames, one for each PDF in the current QA
        qa_tfidf_dfs = calculate_corpus_tfidf(
            all_processed_docs_joined_texts,
            all_docs_stem_to_raw_maps_list,
            qa_doc_indices_global
        )

        # For N-grams and Collocations, it's often more meaningful to analyze
        # the combined text of all documents within a QA, or each doc individually.
        # Here, we'll combine all tokens for the current QA for N-gram/Collocation.
        qa_combined_processed_tokens = []
        qa_combined_stem_to_raw_map = defaultdict(set)  # Aggregate map for the QA

        for global_idx in qa_doc_indices_global:
            qa_combined_processed_tokens.extend(all_docs_processed_tokens_list[global_idx])
            for stem, raw_set in all_docs_stem_to_raw_maps_list[global_idx].items():
                qa_combined_stem_to_raw_map[stem].update(raw_set)

        print(f"Total processed tokens for {qa_name_from_dir} (combined): {len(qa_combined_processed_tokens)}")

        # N-gram and Collocation Analysis for the current QA (using combined tokens)
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

        # --- Save results for the current QA to its own Excel file ---
        # Sanitize qa_name_from_dir for use in filename
        safe_qa_name = "".join(c if c.isalnum() else "_" for c in qa_name_from_dir)
        output_excel_qa = f"keyword_analysis_{safe_qa_name}.xlsx"
        print(f"\n--- Saving results for {qa_name_from_dir} to {output_excel_qa} ---")
        try:
            with pd.ExcelWriter(output_excel_qa, engine='openpyxl') as writer:
                # TF-IDF: Save each PDF's TF-IDF to a separate sheet in the QA's Excel file
                for i, df_doc_tfidf in enumerate(qa_tfidf_dfs):
                    if not df_doc_tfidf.empty:
                        # Get a short name for the sheet (e.g., PDF filename)
                        pdf_original_path = all_docs_original_paths[qa_doc_indices_global[i]]
                        sheet_name_pdf = os.path.basename(pdf_original_path)[:25]  # Max sheet name length ~31
                        # Ensure sheet name is unique if multiple PDFs have same start
                        unique_sheet_name_pdf = sheet_name_pdf
                        count = 1
                        while unique_sheet_name_pdf in writer.sheets:
                            unique_sheet_name_pdf = f"{sheet_name_pdf}_{count}"
                            count += 1
                            if len(unique_sheet_name_pdf) > 30:  # too long
                                unique_sheet_name_pdf = f"Doc{qa_doc_indices_global[i]}"  # fallback
                                break

                        df_doc_tfidf.to_excel(writer, sheet_name=f"TFIDF_{unique_sheet_name_pdf}", index=False)

                if not df_bigrams_qa.empty: df_bigrams_qa.to_excel(writer, sheet_name='QA Top Bigrams', index=False)
                if not df_trigrams_qa.empty: df_trigrams_qa.to_excel(writer, sheet_name='QA Top Trigrams', index=False)
                if not df_colloc_general_qa.empty: df_colloc_general_qa.to_excel(writer,
                                                                                 sheet_name='QA Colloc (General)',
                                                                                 index=False)
                if not df_colloc_seeds_qa.empty: df_colloc_seeds_qa.to_excel(writer, sheet_name='QA Colloc (Seeds)',
                                                                             index=False)
            print(f"Successfully saved results for {qa_name_from_dir} to Excel.")
        except Exception as e:
            print(f"Error saving results for {qa_name_from_dir} to Excel: {e}")

    print("\nCorpus analysis finished.")


if __name__ == "__main__":
    # Optional: Download NLTK resources if not present
    # try: nltk.data.find('corpora/wordnet')
    # except LookupError: nltk.download('wordnet')
    # try: nltk.data.find('corpora/stopwords')
    # except LookupError: nltk.download('stopwords')
    # try: nltk.data.find('tokenizers/punkt')
    # except LookupError: nltk.download('punkt')
    main()