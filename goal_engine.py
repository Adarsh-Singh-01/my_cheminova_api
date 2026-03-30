import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -------- LOAD CAREER DATA --------
with open("data/careers.json", "r") as f:
    careers_data = json.load(f)

career_names = list(careers_data.keys())

# -------- LAZY LOADING MODEL --------
model = None
career_embeddings = None

def get_model():
    global model
    if model is None:
        from sentence_transformers import SentenceTransformer
        print("Loading NLP model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Model loaded ✅")
    return model


# -------- LAZY LOAD EMBEDDINGS --------
def get_career_embeddings():
    global career_embeddings
    if career_embeddings is None:
        print("Generating career embeddings...")
        career_embeddings = get_model().encode(career_names)
        print("Embeddings ready ✅")
    return career_embeddings



# -------- SEMANTIC MATCH FUNCTION --------
def semantic_goal_match(user_query, careers=None, top_k=5):

    # Convert input to embedding
    query_embedding = get_model().encode([user_query])

    # Compare with career embeddings
    similarities = cosine_similarity(
        query_embedding,
        get_career_embeddings()
    )[0]

    # Get top matches
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "career": career_names[idx],
            "score": float(similarities[idx])
        })

    return results