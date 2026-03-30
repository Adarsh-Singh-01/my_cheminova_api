import joblib
import numpy as np

# Load trained model and encoder
model = joblib.load("career_model.pkl")
mlb = joblib.load("skill_encoder.pkl")


def predict_goal_ml(user_skills):

    # Normalize skill names
    user_skills = [skill.strip() for skill in user_skills]

    # Keep only known skills
    valid_skills = [skill for skill in user_skills if skill in mlb.classes_]

    # If no valid skills, return safest fallback
    if not valid_skills:
        return "Business Analyst", 0.2

    user_vector = mlb.transform([valid_skills])

    prediction = model.predict(user_vector)[0]
    probabilities = model.predict_proba(user_vector)[0]

    confidence = float(np.max(probabilities))

    return prediction, confidence
