import json

# Load course database
with open("data/courses.json", "r") as f:
    courses = json.load(f)


def recommend_courses(missing_skills):
    """
    Recommend courses based on missing skills
    """

    recommendations = []

    for skill in missing_skills:
        for course in courses:
            if course["skill"].lower() == skill.lower():
                recommendations.append(course)

    return recommendations
