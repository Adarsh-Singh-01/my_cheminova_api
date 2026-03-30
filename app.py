import json

with open("data/careers.json", "r") as f:
    careers = json.load(f)


from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

from goal_engine import semantic_goal_match
from ml_goal_engine import predict_goal_ml
from course_engine import recommend_courses
from goal_engine import careers_data


app = FastAPI()


from pydantic import BaseModel
from typing import Dict
        
class UserInput(BaseModel):
    skills: Dict[str, int]
    goal: Optional[str] = None
    daily_study_hours: Optional[int] = 2

@app.post("/generate-path")
def generate_path(data: UserInput):

    user_skills = {k.lower(): v for k, v in data.skills.items()}
    goal_input = data.goal

    # -------- HYBRID GOAL PREDICTION --------
    goal_ml, conf_ml = predict_goal_ml(list(user_skills.keys()))

    if goal_input:
        matches = semantic_goal_match(goal_input, top_k=5)
    else:
        skill_text = " ".join(user_skills.keys())
        matches = semantic_goal_match(skill_text, top_k=5)

    goal_nlp = matches[0]["career"]
    conf_nlp = matches[0]["score"]

    if conf_ml > 0.6:
        goal = goal_ml
        confidence = conf_ml
    else:
        goal = goal_nlp
        confidence = conf_nlp

    # -------- REQUIRED SKILLS --------
    required_skills = [skill.lower() for skill in careers_data[goal]["skills"]]

    # -------- WEIGHTED COVERAGE --------
    total_possible_score = len(required_skills) * 5
    achieved_score = 0
    matched_skills = []

    for skill in required_skills:
        if skill in user_skills:
            level = user_skills[skill]
            achieved_score += level
            matched_skills.append(f"{skill} (Level {level})")

    coverage_percentage = round(
        (achieved_score / total_possible_score) * 100, 1
    )

    # -------- SMART SKILL GROUPING --------
    missing_skills = []
    weak_skills = []
    moderate_skills = []

    for skill in required_skills:
        level = user_skills.get(skill, 0)

        if level == 0:
            missing_skills.append(skill)
        elif level <= 2:
            weak_skills.append(skill)
        elif level == 3:
            moderate_skills.append(skill)

    prioritized_skills = missing_skills + weak_skills + moderate_skills

    # -------- LEARNING PATH --------
    path = []

    if prioritized_skills:
        path.append({
            "phase": 1,
            "title": "Foundation / Weak Skills",
            "topics": prioritized_skills[:2]
        })

        if len(prioritized_skills) > 2:
            path.append({
                "phase": 2,
                "title": "Core Skill Improvement",
                "topics": prioritized_skills[2:4]
            })

        if len(prioritized_skills) > 4:
            path.append({
                "phase": 3,
                "title": "Advanced Mastery",
                "topics": prioritized_skills[4:]
            })

    path.append({
        "phase": len(path) + 1,
        "title": "Projects & Portfolio",
        "topics": ["Build 2–3 real-world projects in this domain"]
    })

    # -------- TIMELINE --------
    total_hours = 0

    for skill in required_skills:
        level = user_skills.get(skill, 0)
        total_hours += (5 - level) * 8

    daily_hours = data.daily_study_hours or 2
    estimated_weeks = round((total_hours / daily_hours) / 7, 1)

    timeline = {
        "estimated_total_hours": total_hours,
        "estimated_duration_weeks": estimated_weeks
    }

    # -------- EXPLANATION --------
    if coverage_percentage >= 70:
        level_desc = "strong"
    elif coverage_percentage >= 40:
        level_desc = "moderate"
    else:
        level_desc = "beginner"

    explanation = (
        f"You currently match {coverage_percentage}% of the required skills for {goal}. "
        f"Your background in {', '.join(matched_skills) if matched_skills else 'foundational skills'} "
        f"gives you a {level_desc} starting position."
    )

    courses = recommend_courses(prioritized_skills)

    return {
        "GOAL PREDICTION": {
            "Suggested Goal": goal,
            "Confidence": confidence,
            "Alternatives": matches
        },
        "SKILL ANALYSIS": {
            "Match Percentage": coverage_percentage,
            "Matched Skills": matched_skills,
            "Missing Skills": prioritized_skills,
            "Match Analysis": explanation
        },
        "LEARNING PLAN": {
            "Path": path,
            "Timeline": timeline,
            "Recommended Courses": courses
        }
    }
