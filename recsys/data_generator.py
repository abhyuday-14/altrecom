"""Synthetic-but-realistic student / course / rating generator.

Why synthetic?  Real course-enrolment data is private.  The generator encodes
explicit, documented behavioural rules (interest, career goal, prerequisites,
year-level gating, CGPA vs difficulty) so that experiments are reproducible and
the ground truth is known.  The same CSV schema can be replaced with real data.
"""
import numpy as np
import pandas as pd
from .catalog import COURSES, TOPICS, TOPIC_IDX, CAREERS, BRANCHES

ABILITY_MIN_GPA, ABILITY_MAX_GPA = 5.0, 10.0


def gpa_to_ability(gpa):
    """Map CGPA (5-10) to an 'ability' on the same 1-5 scale as course difficulty."""
    return 1.0 + (np.clip(gpa, ABILITY_MIN_GPA, ABILITY_MAX_GPA) - ABILITY_MIN_GPA) / \
        (ABILITY_MAX_GPA - ABILITY_MIN_GPA) * 4.0


def build_courses_df(rng):
    rows = []
    for cid, name, topic, level, diff, pre, desc in COURSES:
        rows.append(dict(course_id=cid, name=name, topic=TOPICS[topic], level=level,
                         difficulty=diff, credits=3 if diff < 4 else 4,
                         prerequisites=";".join(pre), description=desc))
    return pd.DataFrame(rows)


def generate_dataset(n_students=600, seed=42):
    rng = np.random.default_rng(seed)
    courses = build_courses_df(rng)
    n_items = len(courses)
    topic_idx = courses.topic.map(TOPIC_IDX).values
    level = courses.level.values
    diff = courses.difficulty.values.astype(float)
    id_to_idx = {c: i for i, c in enumerate(courses.course_id)}
    prereq = [[id_to_idx[p] for p in s.split(";") if p] for s in courses.prerequisites]
    quality = rng.normal(0, 0.35, n_items)           # latent course quality (item bias)

    career_names = list(CAREERS)
    branch_names = list(BRANCHES)
    students, ratings = [], []

    for u in range(n_students):
        sid = f"S{u:04d}"
        branch = rng.choice(branch_names, p=[0.4, 0.2, 0.15, 0.25])
        interest = rng.dirichlet(np.array(BRANCHES[branch]) * 0.45)
        interest_n = interest / interest.max()                       # 1.0 for favourite topic
        # career goal correlated with the favourite topics
        career_p = np.array([np.dot(CAREERS[c], interest) for c in career_names]) + 0.03
        career = rng.choice(career_names, p=career_p / career_p.sum())
        career_vec = np.array(CAREERS[career]); career_n = career_vec / career_vec.max()
        gpa = float(np.clip(rng.normal(7.6, 1.0), 5.0, 9.8))
        ability = gpa_to_ability(gpa)
        n_done = int(rng.choice([4, 5, 6], p=[0.3, 0.4, 0.3]))       # completed semesters

        # declared interests = survey answer: top-2 topics, 25% of the time one is random
        top2 = list(np.argsort(-interest)[:2])
        if rng.random() < 0.25:
            top2[1] = int(rng.integers(0, len(TOPICS)))
        declared = ";".join(dict.fromkeys(TOPICS[t] for t in top2))

        students.append(dict(student_id=sid, branch=branch, gpa=round(gpa, 2), career_goal=career,
                             declared_interests=declared, semesters_completed=n_done))

        taken = np.zeros(n_items, bool)
        for sem in range(1, n_done + 1):
            cap = int(np.ceil(sem / 2))
            elig = np.array([(not taken[j]) and level[j] <= cap and all(taken[p] for p in prereq[j])
                             for j in range(n_items)])
            idx = np.where(elig)[0]
            if len(idx) == 0:
                continue
            aff = interest_n[topic_idx[idx]]
            car = career_n[topic_idx[idx]]
            lvl_bonus = np.where(level[idx] == cap, 0.8, np.where(level[idx] == cap - 1, 0.3, 0.0))
            gap = np.maximum(0, diff[idx] - ability)
            util = 3.0 * aff + 1.5 * car + 0.5 * quality[idx] + lvl_bonus - 0.5 * gap
            gumbel = rng.gumbel(size=len(idx))
            k = min(len(idx), int(rng.choice([3, 4, 5], p=[0.3, 0.5, 0.2])))
            chosen = idx[np.argsort(-(util / 1.3 + gumbel))[:k]]
            for j in chosen:
                a = interest_n[topic_idx[j]]; c = career_n[topic_idx[j]]
                r = 2.3 + 2.2 * a + 0.8 * c + quality[j] - 0.45 * max(0, diff[j] - ability) \
                    + rng.normal(0, 0.7)
                ratings.append(dict(student_id=sid, course_id=courses.course_id[j], semester=sem,
                                    rating=int(np.clip(np.round(r), 1, 5))))
            taken[chosen] = True

    return courses, pd.DataFrame(students), pd.DataFrame(ratings)


def save_dataset(courses, students, ratings, folder="data"):
    import os
    os.makedirs(folder, exist_ok=True)
    courses.to_csv(f"{folder}/courses.csv", index=False)
    students.to_csv(f"{folder}/students.csv", index=False)
    ratings.to_csv(f"{folder}/ratings.csv", index=False)


if __name__ == "__main__":
    c, s, r = generate_dataset()
    save_dataset(c, s, r)
    print(len(s), "students", len(c), "courses", len(r), "ratings")
