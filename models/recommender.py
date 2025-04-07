from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_top_k_matches(jd_embeddings, assessment_embeddings, k=2):
    results = []
    for jd in jd_embeddings:
        scores = cosine_similarity([jd], assessment_embeddings)[0]
        top_indices = np.argsort(scores)[-k:][::-1]
        results.append(top_indices)
    return results
