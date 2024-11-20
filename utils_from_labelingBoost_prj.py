import logging
import os
import time

import pandas as pd
import requests

os.environ["OLLAMA_HOST"] = "127.0.0.1:11474"

# Logging
logging.basicConfig(filename="info.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


# Models --------------------------------
def ollama_gen(sys_prompt: str, prompt: str, model: str, params: dict = None, ollama_host: str = None,
               hyperparameters=None):
    '''
        Generate LLM completion with Ollama model
    '''
    from ollama import Client

    # making host
    if ollama_host:
        client = Client(host=ollama_host)
    else:
        client = Client(host=os.environ["OLLAMA_HOST"])

    # making options
    options = hyperparameters or {}
    # NOTE: the max_tokens is not a config, but temperature is.
    if params:
        options.update(params)

    # hardcode temprature 0 everytime
    if 'temperature' not in options:
        options['temperature'] = 0.0

    # print(f'inferencing with hyperparameters: {options}')
    response = client.chat(
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ],
        model=model,
        options=options
    )

    return response, response['message']['content']


def gpt_gen(sys_prompt: str, prompt: str, model: str = "gpt-4o-mini", params: dict = {}):
    from openai import OpenAI

    '''
        Generate LLM completion from OpenAI
    '''
    client = OpenAI()

    resp = client.chat.completions.create(
        model=model,  # gpt4-1106-preview",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        **params
    )
    return resp, resp.choices[0].message.content


def mlops_gen(model: str, sys_prompt: str, usr_prompt: str, retries: int = 5) -> tuple[str, int]:
    """
        Inference using an MLOps model
    """
    headers = {
        'Authorization': f'Bearer {os.environ["MLOPS_API_KEY"]}',
    }

    json_data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": sys_prompt
            },
            {
                "role": "user",
                "content": usr_prompt
            }
        ]
    }

    url = "http://mlops.huawei.com/mlops-service/api/v1/agentService/v1/chat/completions"
    for i in range(retries):
        response = requests.post(url, headers=headers, json=json_data, stream=False)
        try:
            resp_json = response.json()
            content = resp_json['choices'][0]['message']['content']
            break
        except Exception as e:
            # print(f"[MLOPS] failed to generate due to: {str(e)}\n, response received: {json.dumps(response, indent=4)}")
            print(f"^^ [MLOPS] failed to generate; retry #{i + 1} MLOps gen after 5 second delay ^^")
            time.sleep(5)
            continue  # retry MLOps inference
    else:
        print(f"[MLOPS] failed to generate; \n, response received: {response.__dict__}")
        raise Exception

    return content, response['code']


# load diff models:
def load_code_t5_model(device='cuda'):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    model_name = "Salesforce/codet5p-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    print(f"Loaded model: {model_name}")  # takes like 1.5 minutes

    return model, tokenizer


# === experiment with different embeddings

def get_starencoder_embeddings(text_corpus, hf_token=None, model=None, tokenizer=None):
    from transformers import AutoTokenizer, AutoModel
    import torch

    if hf_token is None:
        # there's some issue where I gotta provide this token everytime to access starcoder
        # not wasting time debugging now rn, got too much on my plate.
        hf_token = "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"

    # Load tokenizer and model
    if model is None:
        tokenizer = AutoTokenizer.from_pretrained('bigcode/starencoder', token=hf_token)
        model = AutoModel.from_pretrained('bigcode/starencoder', token=hf_token)

    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use end-of-sequence token if no padding token

    # Batching the process coz GPU runs out of memory
    inputs = tokenizer(
        text_corpus,
        padding=True,
        truncation=True,
        max_length=1024,  # used max offered by starcoder
        return_tensors="pt"
    )

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling

    return embeddings.numpy()


def get_codet5_embeddings(text_corpus, model=None, tokenizer=None, device='cuda'):
    import torch, gc
    import numpy as np

    if model is None:
        model, tokenizer = load_code_t5_model(device)

    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use end-of-sequence token if no padding token

    # Dynamically get max_length from the tokenizer
    max_length = min(tokenizer.model_max_length, 1024)  # Cap max_length at 1024

    all_embeddings = []
    # these vals work on a free GPU (40GB)
    batch_size = 16
    max_length = 2048

    # Iterate over smaller batches
    for i in range(0, len(text_corpus), batch_size):
        batch_text = text_corpus[i:i + batch_size]

        # Tokenize with truncation and padding
        inputs = tokenizer(
            batch_text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            # Forward pass through encoder
            encoder_outputs = model.encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

            # Get mean pooled embeddings (move to CPU to save GPU memory)
            embeddings = encoder_outputs.last_hidden_state.mean(dim=1).cpu()
            embedding_numpy = embeddings.detach().cpu().numpy()
            all_embeddings.append(embedding_numpy)

        # Clear GPU memory after each batch
        del inputs, encoder_outputs, embeddings
        torch.cuda.empty_cache()
        gc.collect()
        print(f"GPU Memory Usage: {torch.cuda.memory_allocated() / 1e6:.2f} MB")

    # Concatenate all NumPy arrays into a single array
    final_embeddings = np.concatenate(all_embeddings, axis=0)

    assert len(text_corpus) == final_embeddings.shape[0], "Embeddings length does not match input length"

    return final_embeddings


# ==== Semantic diversity
def get_semantic_diversity(text_corpus, embeddings=None):
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

    if embeddings is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2", trust_remote_code=True)
        embeddings = model.encode(text_corpus, show_progress_bar=True)

    assert len(embeddings) == len(text_corpus), "Embeddings length does not match input length"

    cosine_sim_matrix = cosine_similarity(embeddings)

    # Exclude self-similarity by setting to 0
    np.fill_diagonal(cosine_sim_matrix, 0)

    # Compute row-wise median similarity, ignoring NaN
    row_median_similarity = np.median(cosine_sim_matrix, axis=1)
    row_median_similarity = np.nan_to_num(row_median_similarity, nan=0.0)  # Handle edge cases fill na to 0

    # Compute dissimilarity as 1 - median similarity
    row_dissimilarity = 1 - row_median_similarity

    # Scaling the values from 0 to 1
    scaler = MinMaxScaler()
    scaled_dissimilarity = scaler.fit_transform(row_dissimilarity.reshape(-1, 1))

    return scaled_dissimilarity.flatten(), row_dissimilarity.tolist()


# ==== Shapley stuff
def do_embedding_shapley(df_sub, x_col, y_col, _type, model=None, tokenizer=None):
    from sklearn.model_selection import train_test_split
    import numpy as np
    from valda.valuation import DataValuation

    # we preserve indices!
    preserved_ind = df_sub.index.to_list()

    def get_embeddings(X, y, preserved_ind):

        # Perform the split on both X and sending indices to keep track of original locations
        trnX, devX, trnY, devY, trn_indices, _ = train_test_split(
            X, y, preserved_ind, test_size=0.2, random_state=42
        )

        # Use the embedding model to encode the texts
        if _type == 'starencoder':
            trainX_vect = get_starencoder_embeddings(text_corpus=trnX.tolist(), model=model, tokenizer=tokenizer)
            devX_vect = get_starencoder_embeddings(text_corpus=devX.tolist(), model=model, tokenizer=tokenizer)
        elif _type == 'codet5':
            trainX_vect = get_codet5_embeddings(trnX.tolist(), model, tokenizer)
            devX_vect = get_codet5_embeddings(devX.tolist(), model, tokenizer)
        elif _type == 'paraphrase-mpnet':
            trainX_vect = model.encode(trnX)
            devX_vect = model.encode(devX)
        else:
            raise ValueError("Invalid model type")

        # Explicitly convert to NumPy arrays for shapley
        trainX_vect = np.array(trainX_vect)
        devX_vect = np.array(devX_vect)
        trnY = np.array(trnY)
        devY = np.array(devY)

        return trainX_vect, devX_vect, trnY, devY, trn_indices

    def _do_shapley(trainX_vect, trnY, devX_vect, devY):
        # Define a DataValuation instance
        dv = DataValuation(trainX_vect, trnY, devX_vect, devY)

        # with default logistic regression is fastest!
        vals = dv.estimate(method='cs-shapley')
        return vals

    # Get TF-IDF vectors and other data
    X = df_sub[x_col].values
    y = df_sub[y_col].values

    trainX_vect, devX_vect, trnY, devY, trn_indices = get_embeddings(X, y, preserved_ind)
    vals = _do_shapley(trainX_vect, trnY, devX_vect, devY)

    # vals indexes are in 0 to n_sample, we need to map them back to the original indices
    mapped_vals = {trn_indices[i]: vals[i] for i in range(len(vals))}

    # Attach the shapley values back to the original sampled df
    df_sub['shapley_vals'] = df_sub.index.map(mapped_vals)

    # Return the sampled df with shapley values
    # this will drop the test set!
    df_sub = df_sub.dropna(subset=['shapley_vals'])

    # validation
    print(len(df_sub) == len(trn_indices))

    return df_sub


def do_shapley(df, n_sample, y_label):
    '''
    depreciated: this is for TF-IDF, which is too primitive.
    upgraded to `do_embedding_shapley` that uses `nomic-embed-text-v1` Sentence Transformers
    '''
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    import numpy as np
    import re
    from valda.valuation import DataValuation

    # Sample the dataframe
    df_sub = df.sample(n_sample, random_state=42)

    # we preserve indices!
    preserved_ind = df_sub.index.to_list()

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'[_\W]+', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def get_tfidf(X, y, preserved_ind):
        # Perform the split on both X and sending indices to keep track of original locations
        trnX, devX, trnY, devY, trn_indices, _ = train_test_split(
            X, y, preserved_ind, test_size=0.2, random_state=42
        )

        vectorizer = TfidfVectorizer()

        # Fit the vectorizer on the training data and transform both training and development sets
        trainX_vect = vectorizer.fit_transform(trnX)
        devX_vect = vectorizer.transform(devX)

        # Explicitly convert to NumPy arrays for shapley
        trainX_vect = trainX_vect.toarray()
        devX_vect = devX_vect.toarray()
        trnY = np.array(trnY)
        devY = np.array(devY)

        return trainX_vect, devX_vect, trnY, devY, trn_indices

    def _do_shapley(trainX_vect, trnY, devX_vect, devY):
        # Define a DataValuation instance
        dv = DataValuation(trainX_vect, trnY, devX_vect, devY)

        # with default logistic regression is fastest!
        vals = dv.estimate(method='cs-shapley')
        return vals

    # Clean all the texts
    df_sub['commit_msg_clean'] = df_sub['commit_msg'].apply(clean_text)
    X = df_sub['commit_msg_clean'].tolist()
    y = df_sub[y_label].tolist()

    # Get TF-IDF vectors and other data
    trainX_vect, devX_vect, trnY, devY, trn_indices = get_tfidf(X, y, preserved_ind)
    vals = _do_shapley(trainX_vect, trnY, devX_vect, devY)

    # vals indexes are in 0 to n_sample, we need to map them back to the original indices
    mapped_vals = {trn_indices[i]: vals[i] for i in range(len(vals))}

    # Attach the shapley values back to the original sampled df
    df_sub['shapley_vals'] = df_sub.index.map(mapped_vals)

    # Return the sampled df with shapley values
    # this will drop the test set!
    df_sub = df_sub.dropna(subset=['shapley_vals'])

    print(len(df_sub) == len(trn_indices))
    return df_sub


# === results evaluation

def fetch_results(results_df, model_label, correct_label='human_fix', fix_model_label=True):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

    if fix_model_label:
        results_df[model_label] = results_df[model_label].apply(lambda x: 1 if x == 'Y' else (0 if x == 'N' else -1))
        print('removed points', len(results_df[results_df[model_label] == -1]))
        results_df = results_df[results_df[model_label] != -1].copy()

        # model label is int, correct label should also be int
        if results_df[correct_label].dtype == 'object':
            results_df[correct_label] = results_df[correct_label].apply(
                lambda x: 1 if x == 'Y' else (0 if x == 'N' else -1))

    else:
        results_df[model_label] = results_df[model_label]
        results_df[correct_label] = results_df[correct_label]

    accuracy = accuracy_score(results_df[correct_label], results_df[model_label])
    precision = precision_score(results_df[correct_label], results_df[model_label])
    recall = recall_score(results_df[correct_label], results_df[model_label])
    f1 = f1_score(results_df[correct_label], results_df[model_label])
    mcc = matthews_corrcoef(results_df[correct_label], results_df[model_label])

    # Calculate the metrics and add them to the summary dataframe
    accuracy = round(accuracy * 100, 2)
    precision = float(round(precision * 100, 2))
    recall = float(round(recall * 100, 2))
    f1 = float(round(f1 * 100, 2))
    mcc = float(round(mcc * 100, 2))

    print(f"{accuracy=}, {precision=}, {recall=}, {f1=}, {mcc=}")

    return accuracy, precision, recall, f1, mcc


# stats stuff

def get_anova(_df, category_col, num_col):
    """
    Depreciated:
        use corr_bw_numerical_categorical instead
        that does both anova and eta squared (effect size)
    """
    from scipy import stats

    _df[category_col] = _df[category_col].astype('category')

    # ANOVA: for multiple categories
    grouped_data = [_df[_df[category_col] == cat][num_col]
                    for cat in _df[category_col].unique()]

    f_stat, p_value_anova = stats.f_oneway(*grouped_data)
    print(f"ANOVA F-statistic: {f_stat}, p-value: {p_value_anova}")
    return f_stat, p_value_anova


def corr_bw_two_categorical_cols(df, col1, col2):
    from scipy.stats import chi2_contingency
    import numpy as np

    # Create contingency table
    contingency_table = pd.crosstab(df[col1], df[col2])

    # Perform Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    # Return results as a dictionary
    print(f'{chi2=}, {p=}, {dof=}, {expected=}')

    if p < 0.05:
        print('chi sq indicates statistical significance, calculating the amount of correlation via cramers V')

        # Calculate Cramér's V
        n = contingency_table.sum().sum()  # total sample size
        min_dim = min(contingency_table.shape) - 1  # min(rows - 1, columns - 1)
        cramers_v = np.sqrt(chi2 / (n * min_dim))
        print(f'{cramers_v=}')

        """ Cramer's V interpretation:
        0.0 - 0.1: Negligible association
        0.1 - 0.3: Weak association
        0.3 - 0.5: Moderate association
        > 0.5: Strong association
        """

    return {
        "chi_square_statistic": chi2,
        "p_value": p,
        "degrees_of_freedom": dof,
        "expected_frequencies": expected
    }


def corr_bw_numerical_categorical(data, numerical_column, categorical_column, p_val_thresh=0.05):
    import pingouin as pg

    # Perform ANOVA
    anova_results = pg.anova(data=data, dv=numerical_column, between=categorical_column)
    print(f'{anova_results=}')

    # Calculate effect size only if statistically significant:
    p_value = anova_results['p-unc'][0]
    print(f'{p_value=}')

    if p_value < p_val_thresh:
        eta_squared = anova_results['np2'][0]  # Partial eta squared
        print(f'effect size is {eta_squared}')

        # Interpretation based on thresholds
        if eta_squared < 0.01:
            interpretation = "small"
        elif 0.01 <= eta_squared < 0.06:
            interpretation = "medium"
        else:
            interpretation = "large"

        print('''
            Small effect: η² < 0.01
            Medium effect: 0.01 ≤ η² < 0.06
            Large effect: η² ≥ 0.14
            ''')

        # Print results
        print(f"For Effect Size (η²): {eta_squared:.4f} - Interpretation: {interpretation.capitalize()} effect")

        return anova_results, eta_squared, interpretation

    return anova_results, None, None


def numerical_correlation_analysis_dendrogram(df):
    # TODO: untested, just a code template.
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    correlations = df.corr()
    dissimilarity = 1 - abs(correlations)
    Z = linkage(squareform(dissimilarity), 'complete')

    fig, ax = plt.subplots()
    dendrogram(Z, labels=df.columns, orientation='top', leaf_rotation=90);
    fig.tight_layout()
    fig.savefig('{}.pdf', dpi=3500)  # TODO: fix
