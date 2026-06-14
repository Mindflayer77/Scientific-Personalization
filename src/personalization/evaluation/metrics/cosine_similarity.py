import pandas as pd
import numpy as np
from sentence_transformers import util
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
import umap

def analyze_semantic_similarity(df, model, text_col='text', user_col='user_id'):
    """
    Computes Intra-User and Inter-User similarity broken down by specific User Types.
    """
    print("Encoding texts...")
    embeddings = model.encode(df[text_col].tolist(), convert_to_tensor=True, show_progress_bar=True)
    
    print("Computing similarity matrix...")
    sim_matrix = util.cos_sim(embeddings, embeddings)
    
    n_rows = len(df)
    
    print("Getting users")
    users = df[user_col].values 

    print("Build indices")
    i_idx, j_idx = np.triu_indices(n_rows, k=1)

    print("Get scores")
    scores = sim_matrix[i_idx, j_idx]        
    
    print("Same user mask")
    same_mask = users[i_idx] == users[j_idx]   

    print("Intra df")
    intra = pd.DataFrame({
        "user_type_a":  users[i_idx[same_mask]],
        "user_type_b":  users[i_idx[same_mask]],
        "category":   "Intra-User (Same)",
        "similarity": scores[same_mask].detach().cpu(),
    })

    print("Different user mask")
    inter_idx = ~same_mask
    inter = pd.concat([
        pd.DataFrame({"user_type_a": users[i_idx[inter_idx]],
                    "user_type_b": users[j_idx[inter_idx]],
                    "category": "Inter-User (Different)",
                    "similarity": scores[inter_idx].detach().cpu()}),
    ])

    res_df = pd.concat([intra, inter], ignore_index=True)
    summary_agg = res_df.groupby(['user_type_a', 'category'])['similarity'].agg(['mean', 'std', 'count']).reset_index().rename({'user_type_a': 'user_type'}, axis=1)

    res_df["pair_key"] = res_df.apply(lambda r: frozenset([r["user_type_a"], r["user_type_b"]]), axis=1)
    summary_pairs = res_df.groupby(['pair_key'])['similarity'].agg(['mean', 'std', 'count']).reset_index()

    return summary_pairs, summary_agg, embeddings

def calculate_umap(df, embeddings, n_neighbours=10, random_state=42):
    labels = df["user_id"].values
    label_ids = LabelEncoder().fit_transform(labels)

    score = silhouette_score(embeddings, label_ids)

    embedding_2d = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=n_neighbours).fit_transform(embeddings)

    return embedding_2d, score, labels 