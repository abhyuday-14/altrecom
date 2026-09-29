"""Data container, eligibility rules, user context and temporal train/test split."""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from .catalog import TOPICS, TOPIC_IDX, CAREERS


class CourseData:
    def __init__(self, courses, students, ratings):
        self.courses = courses.reset_index(drop=True)
        self.students = students.reset_index(drop=True)
        self.ratings = ratings
        self.item_ids = list(self.courses.course_id)
        self.item_idx = {c: i for i, c in enumerate(self.item_ids)}
        self.user_ids = list(self.students.student_id)
        self.user_idx = {s: i for i, s in enumerate(self.user_ids)}
        self.n_users, self.n_items = len(self.user_ids), len(self.item_ids)
        self.level = self.courses.level.values
        self.difficulty = self.courses.difficulty.values.astype(float)
        self.topic = self.courses.topic.map(TOPIC_IDX).values
        # prerequisite matrix P[j, p] = 1 if p is a prerequisite of j
        self.P = np.zeros((self.n_items, self.n_items))
        for j, s in enumerate(self.courses.prerequisites.fillna("")):
            for p in [x for x in s.split(";") if x]:
                self.P[j, self.item_idx[p]] = 1
        self.n_prereq = self.P.sum(1)
        self.text = (self.courses.name + " " + self.courses.description + " " +
                     self.courses.topic.str.replace("/", "")).tolist()

    def matrix(self, df):
        """Dense user x item rating matrix (0 = not rated)."""
        R = np.zeros((self.n_users, self.n_items))
        u = df.student_id.map(self.user_idx).values
        i = df.course_id.map(self.item_idx).values
        R[u, i] = df.rating.values
        return R

    def eligible(self, taken, target_semester):
        """Not taken, within the year-level cap, and all prerequisites completed."""
        cap = int(np.ceil(target_semester / 2))
        prereq_ok = (self.P @ taken.astype(float)) >= self.n_prereq
        return (~taken) & (self.level <= cap) & prereq_ok


@dataclass
class UserContext:
    ratings: np.ndarray            # length n_items, 0 = unrated
    taken: np.ndarray              # bool, courses already completed
    gpa: float
    career: str
    interests: list = field(default_factory=list)   # list of topic names
    next_semester: int = 5
    self_index: int = -1           # row in training matrix (excluded from neighbours)


def make_context(data, R, student_id, next_semester=None, taken=None, ratings=None):
    u = data.user_idx[student_id]
    row = data.students.iloc[u]
    ratings = R[u] if ratings is None else ratings
    taken = (R[u] > 0) if taken is None else taken
    nxt = int(row.semesters_completed) + 1 if next_semester is None else next_semester
    interests = [t for t in str(row.declared_interests).split(";") if t]
    return UserContext(ratings, taken, float(row.gpa), row.career_goal, interests, nxt, u)


def temporal_split(data, drop_last=0):
    """Leave-last-semester-out. drop_last=1 shifts everything back one semester
    (used only for validation / hyper-parameter tuning, so the test set stays untouched)."""
    df = data.ratings.merge(data.students[["student_id", "semesters_completed"]], on="student_id")
    df["last"] = df.semesters_completed - drop_last
    df = df[df.semester <= df["last"]]
    train = df[df.semester < df["last"]][["student_id", "course_id", "semester", "rating"]]
    test = df[df.semester == df["last"]][["student_id", "course_id", "semester", "rating"]]
    target_sem = (data.students.set_index("student_id").semesters_completed - drop_last).to_dict()
    return train, test, target_sem
