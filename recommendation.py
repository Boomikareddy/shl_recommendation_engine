import pandas as pd
from preprocessing.text_cleaning import clean_text
from utils.embeddings import get_embedding
from models.recommender import get_top_k_matches

def get_recommendations(query, k=2):
    # Load data
    jds = pd.read_csv("data/sample_jds.csv")
    assessments = pd.read_csv("data/sample_assessments.csv")

    # Preprocess
    jds['processed'] = jds['description'].apply(clean_text)
    assessments['processed'] = assessments['description'].apply(clean_text)

    # Get embeddings
    jd_embeddings = jds['processed'].apply(get_embedding).tolist()
    assessment_embeddings = assessments['processed'].apply(get_embedding).tolist()

    # Add query
    query_clean = clean_text(query)
    query_embedding = get_embedding(query_clean)

    # Find matches for query (instead of looping through jds)
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity([query_embedding], assessment_embeddings)[0]

    top_indices = similarities.argsort()[-k:][::-1]

    recommendations = []
    for idx in top_indices:
        assess_id = assessments.iloc[idx]['id']
        assess_name = assessments.iloc[idx]['name']
        recommendations.append({'id': assess_id, 'name': assess_name})

    return recommendations
