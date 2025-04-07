import pandas as pd
from preprocessing.text_cleaning import clean_text
from utils.embeddings import get_embedding
from models.recommender import get_top_k_matches

# Load data
jds = pd.read_csv("data/sample_jds.csv")
assessments = pd.read_csv("data/sample_assessments.csv")

# Preprocess
jds['processed'] = jds['description'].apply(clean_text)
assessments['processed'] = assessments['description'].apply(clean_text)

# Get embeddings
jd_embeddings = jds['processed'].apply(get_embedding).tolist()
assessment_embeddings = assessments['processed'].apply(get_embedding).tolist()

# Get recommendations
top_k = get_top_k_matches(jd_embeddings, assessment_embeddings, k=2)

# Save results
recommendations = []
for i, indices in enumerate(top_k):
    jd_id = jds.iloc[i]['id']
    jd_title = jds.iloc[i]['title']
    for idx in indices:
        assess_id = assessments.iloc[idx]['id']
        assess_name = assessments.iloc[idx]['name']
        recommendations.append([jd_id, jd_title, assess_id, assess_name])

output = pd.DataFrame(recommendations, columns=["JD_ID", "JD_Title", "Assessment_ID", "Assessment_Name"])
output.to_csv("results/output_recommendations.csv", index=False)

print("✅ Recommendations generated and saved in results/output_recommendations.csv")
