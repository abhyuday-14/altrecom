"""
Personalized Course Recommendation System  -  CS24308 Artificial Intelligence project
=====================================================================================
Single-file submission. Compares Random, Popularity, Content-Based (TF-IDF), User-kNN
collaborative filtering and Matrix Factorization (ALS) against our own improvement,
the Hybrid PCHR (Prerequisite-, Career- and difficulty-aware Hybrid with MMR re-ranking).

Requirements : pip install numpy pandas scikit-learn matplotlib

Usage
    python course_recommender.py compare                 # all result tables + results/comparison.png
    python course_recommender.py compare --seeds 2       # quicker run
    python course_recommender.py demo --student S0007    # every model's top-5 for one student
    python course_recommender.py demo --new              # cold-start: type your own profile
    python course_recommender.py generate                # write data/*.csv

Sections
    1 catalogue   2 data generator   3 dataset and split   4 models   5 evaluation   6 command line
"""
import argparse
import itertools
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================================================
# 1. DOMAIN KNOWLEDGE: topics, careers and the 48-course catalogue
# ============================================================================================
TOPICS = ["AI/ML", "Data", "Systems", "Web/Software", "Security", "Theory"]
TOPIC_IDX = {t: i for i, t in enumerate(TOPICS)}

# How strongly each career goal relates to every topic (rows follow TOPICS order).
CAREERS = {
    "Data Scientist":          [0.30, 0.45, 0.00, 0.05, 0.00, 0.20],
    "ML Engineer":             [0.55, 0.20, 0.10, 0.10, 0.00, 0.05],
    "Web/App Developer":       [0.00, 0.10, 0.05, 0.75, 0.10, 0.00],
    "Security Analyst":        [0.00, 0.00, 0.25, 0.10, 0.60, 0.05],
    "Systems Engineer":        [0.00, 0.05, 0.65, 0.15, 0.10, 0.05],
    "Research / Higher Studies": [0.30, 0.05, 0.10, 0.00, 0.05, 0.50],
}

# Dirichlet prior tilt per branch (cosmetic realism + gives CF something to learn).
BRANCHES = {
    "CSE":     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "IT":      [0.8, 1.0, 0.8, 1.6, 1.0, 0.6],
    "ECE":     [0.6, 0.6, 1.8, 0.8, 1.0, 0.8],
    "AI&DS":   [1.8, 1.6, 0.6, 0.8, 0.5, 0.8],
}

COURSES = [
    # ---- Year 1 ----
    ("CS101", "Programming with Python", 3, 1, 2, [], "python programming variables loops functions scripting problem solving"),
    ("CS102", "Discrete Mathematics", 5, 1, 3, [], "logic sets relations graphs combinatorics proofs"),
    ("CS103", "Linear Algebra for Computing", 5, 1, 3, [], "vectors matrices eigenvalues linear transformations applications"),
    ("CS104", "Probability and Statistics", 1, 1, 3, [], "probability random variables distributions hypothesis testing inference"),
    ("CS105", "Digital Logic Design", 2, 1, 2, [], "boolean algebra logic gates circuits flip flops"),
    ("CS106", "Web Fundamentals", 3, 1, 2, [], "html css javascript responsive web pages"),
    ("CS107", "Introduction to Data Analysis", 1, 1, 2, [], "spreadsheets data cleaning visualization charts exploratory analysis"),
    ("CS108", "Cyber Hygiene and Ethics", 4, 1, 1, [], "passwords privacy phishing safe computing ethics"),
    # ---- Year 2 ----
    ("CS201", "Data Structures", 3, 2, 3, ["CS101"], "arrays linked lists trees graphs hashing algorithms complexity"),
    ("CS202", "Database Management Systems", 1, 2, 3, ["CS101"], "sql relational model normalization transactions indexing"),
    ("CS203", "Computer Organization", 2, 2, 3, ["CS105"], "processor architecture instruction set memory hierarchy pipelining"),
    ("CS204", "Object Oriented Programming", 3, 2, 2, ["CS101"], "classes inheritance polymorphism design patterns java"),
    ("CS205", "Design and Analysis of Algorithms", 5, 2, 4, ["CS201", "CS102"], "divide conquer dynamic programming greedy graph algorithms np completeness"),
    ("CS206", "Python for Data Science", 1, 2, 2, ["CS101", "CS107"], "numpy pandas matplotlib data wrangling notebooks"),
    ("CS207", "Introduction to Artificial Intelligence", 0, 2, 3, ["CS101"], "search heuristics knowledge representation agents constraint satisfaction"),
    ("CS208", "Computer Networks", 2, 2, 3, ["CS105"], "tcp ip routing protocols layers sockets"),
    ("CS209", "Web Development with JavaScript", 3, 2, 2, ["CS106"], "javascript dom react frontend apis"),
    ("CS210", "Introduction to Cryptography", 4, 2, 3, ["CS102"], "encryption symmetric public key hashing signatures number theory"),
    ("CS211", "Statistical Methods for Data Analysis", 1, 2, 3, ["CS104"], "regression anova sampling confidence intervals statistical modelling"),
    ("CS212", "Theory of Computation", 5, 2, 4, ["CS102"], "automata regular languages turing machines decidability"),
    # ---- Year 3 ----
    ("CS301", "Machine Learning", 0, 3, 4, ["CS103", "CS104", "CS206"], "supervised unsupervised learning regression classification clustering model evaluation"),
    ("CS302", "Deep Learning", 0, 3, 5, ["CS301"], "neural networks backpropagation cnn rnn optimization pytorch"),
    ("CS303", "Natural Language Processing", 0, 3, 4, ["CS301"], "text tokenization embeddings language models sentiment translation"),
    ("CS304", "Computer Vision", 0, 3, 4, ["CS301"], "image processing convolution object detection segmentation"),
    ("CS305", "Operating Systems", 2, 3, 4, ["CS201", "CS203"], "processes threads scheduling memory management file systems deadlocks"),
    ("CS306", "Data Mining and Warehousing", 1, 3, 3, ["CS202", "CS206"], "association rules clustering classification olap data warehouse etl"),
    ("CS307", "Big Data Analytics", 1, 3, 4, ["CS202", "CS206"], "hadoop spark mapreduce distributed processing streaming"),
    ("CS308", "Data Visualization", 1, 3, 2, ["CS206"], "dashboards storytelling tableau plotting visual analytics"),
    ("CS309", "Software Engineering", 3, 3, 2, ["CS204"], "requirements agile testing uml version control project management"),
    ("CS310", "Full Stack Web Development", 3, 3, 3, ["CS209", "CS202"], "node express mongodb rest api authentication deployment"),
    ("CS311", "Mobile App Development", 3, 3, 3, ["CS204"], "android flutter user interface sensors app store"),
    ("CS312", "Network Security", 4, 3, 4, ["CS208", "CS210"], "firewalls intrusion detection vpn ssl attacks defense"),
    ("CS313", "Ethical Hacking and Penetration Testing", 4, 3, 4, ["CS208"], "vulnerability scanning exploits metasploit reconnaissance web attacks"),
    ("CS314", "Distributed Systems", 2, 3, 5, ["CS208", "CS201"], "replication consensus fault tolerance clocks microservices"),
    ("CS315", "Compiler Design", 5, 3, 5, ["CS212", "CS201"], "lexical analysis parsing code generation optimization"),
    ("CS316", "Cloud Computing", 2, 3, 3, ["CS208"], "virtualization aws containers docker serverless scalability"),
    # ---- Year 4 ----
    ("CS401", "Reinforcement Learning", 0, 4, 5, ["CS301"], "agents rewards markov decision processes q learning policy gradient"),
    ("CS402", "Generative AI and LLMs", 0, 4, 5, ["CS302"], "transformers large language models prompting diffusion fine tuning"),
    ("CS403", "Recommender Systems", 0, 4, 4, ["CS301"], "collaborative filtering content based matrix factorization ranking evaluation"),
    ("CS404", "MLOps and Model Deployment", 0, 4, 3, ["CS301"], "model deployment monitoring pipelines docker versioning"),
    ("CS405", "Time Series Forecasting", 1, 4, 4, ["CS211"], "arima seasonality forecasting trends prediction"),
    ("CS406", "Information Retrieval", 1, 4, 3, ["CS202"], "indexing search engines ranking relevance query"),
    ("CS407", "Cyber Forensics", 4, 4, 3, ["CS208", "CS108"], "digital evidence disk analysis incident response logs"),
    ("CS408", "Blockchain Technology", 4, 4, 3, ["CS210"], "distributed ledger consensus smart contracts ethereum"),
    ("CS409", "High Performance Computing", 2, 4, 5, ["CS305"], "parallel programming gpu cuda mpi performance"),
    ("CS410", "Embedded and IoT Systems", 2, 4, 3, ["CS203"], "microcontrollers sensors arduino iot protocols real time"),
    ("CS411", "Advanced Algorithms", 5, 4, 5, ["CS205"], "randomized approximation algorithms network flow amortized analysis"),
    ("CS412", "Quantum Computing", 5, 4, 5, ["CS103", "CS212"], "qubits quantum gates superposition algorithms shor grover"),
]


# ============================================================================================
# 2. DATASET: synthetic students, choices and ratings
# ============================================================================================
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


# ============================================================================================
# 3. DATA CONTAINER, ELIGIBILITY RULE AND TEMPORAL SPLIT
# ============================================================================================
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


# ============================================================================================
# 4. RECOMMENDATION MODELS (Random, Popularity, Content, kNN, MF, Hybrid PCHR)
# ============================================================================================
def zscore(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)


class BaseRecommender:
    name = "base"

    def fit(self, data, R):
        self.data, self.R = data, R
        self.counts = (R > 0).sum(0).astype(float)
        return self

    def predict(self, ctx):
        """Predicted rating (1-5) for every course, or None if the model has no rating scale."""
        return None

    def scores(self, ctx):
        raise NotImplementedError

    def rank(self, ctx, k, allowed):
        s = self.scores(ctx) + 1e-9 * self.counts       # popularity breaks ties (cold start)
        s = np.where(allowed, s, -np.inf)
        return [int(j) for j in np.argsort(-s)[:k] if np.isfinite(s[j])]


class RandomRecommender(BaseRecommender):
    """Sanity-check floor: any useful model must beat random ranking."""
    name = "Random"

    def fit(self, data, R):
        super().fit(data, R)
        self.rng = np.random.default_rng(0)
        return self

    def scores(self, ctx):
        return self.rng.random(self.data.n_items)

    def rank(self, ctx, k, allowed):
        s = np.where(allowed, self.scores(ctx), -np.inf)
        return [int(j) for j in np.argsort(-s)[:k] if np.isfinite(s[j])]


class PopularityRecommender(BaseRecommender):
    name = "Popularity"

    def fit(self, data, R):
        super().fit(data, R)
        m = R > 0
        self.gmean = R[m].mean()
        self.item_mean = (R.sum(0) + 5 * self.gmean) / (m.sum(0) + 5)   # damped mean
        return self

    def scores(self, ctx):
        return self.counts

    def predict(self, ctx):
        return self.item_mean


class ContentBasedRecommender(BaseRecommender):
    name = "Content-Based (TF-IDF)"

    def fit(self, data, R):
        super().fit(data, R)
        X = TfidfVectorizer().fit_transform(data.text)
        self.S = cosine_similarity(X)                    # item-item similarity
        return self

    def scores(self, ctx):
        w = np.maximum(ctx.ratings - 2.0, 0)             # liked courses carry weight
        if w.sum() == 0:
            return np.zeros(self.data.n_items)
        return (w @ self.S) / w.sum()


class UserKNNRecommender(BaseRecommender):
    name = "User-kNN CF"

    def __init__(self, k=30, shrink=2.0):
        self.k, self.shrink = k, shrink

    def fit(self, data, R):
        super().fit(data, R)
        self.M = (R > 0).astype(float)
        n = np.maximum(self.M.sum(1), 1)
        self.mu = R.sum(1) / n
        self.C = (R - self.mu[:, None]) * self.M          # mean-centred ratings
        self.Cn = np.linalg.norm(self.C, axis=1) + 1e-9
        return self

    def predict(self, ctx):
        r = ctx.ratings; m = r > 0
        if m.sum() == 0:
            return np.full(self.data.n_items, self.R[self.R > 0].mean())
        mu = r[m].mean()
        c = (r - mu) * m
        sims = (self.C @ c) / (self.Cn * (np.linalg.norm(c) + 1e-9))
        if ctx.self_index >= 0:
            sims[ctx.self_index] = -1
        top = np.argsort(-sims)[:self.k]
        top = top[sims[top] > 0]
        w = sims[top]
        num = w @ self.C[top]
        den = w @ self.M[top] + self.shrink              # shrinkage: few raters -> stay near mean
        return np.clip(mu + num / den, 1, 5)

    def scores(self, ctx):
        return self.predict(ctx)


class MatrixFactorization(BaseRecommender):
    """Biased matrix factorisation r_ui ~ mu + b_i + b_u + p_u . q_i trained with ALS.
    New / held-out students are 'folded in' by solving a ridge regression against item factors."""
    name = "Matrix Factorization (ALS)"

    def __init__(self, k=8, reg=4.0, iters=15, seed=0):
        self.k, self.reg, self.iters, self.seed = k, reg, iters, seed

    def fit(self, data, R):
        super().fit(data, R)
        M = R > 0
        self.mu = R[M].mean()
        self.bi = ((R - self.mu) * M).sum(0) / (M.sum(0) + 5)
        bu = ((R - self.mu - self.bi) * M).sum(1) / (M.sum(1) + 5)
        E = (R - self.mu - self.bi - bu[:, None]) * M
        rng = np.random.default_rng(self.seed)
        n_u, n_i = R.shape
        P = rng.normal(0, 0.1, (n_u, self.k)); Q = rng.normal(0, 0.1, (n_i, self.k))
        I = np.eye(self.k) * self.reg
        for _ in range(self.iters):
            for u in range(n_u):
                idx = np.where(M[u])[0]
                if len(idx):
                    Qi = Q[idx]; P[u] = np.linalg.solve(Qi.T @ Qi + I, Qi.T @ E[u, idx])
            for i in range(n_i):
                idx = np.where(M[:, i])[0]
                if len(idx):
                    Pu = P[idx]; Q[i] = np.linalg.solve(Pu.T @ Pu + I, Pu.T @ E[idx, i])
        self.Q = Q
        return self

    def predict(self, ctx):
        r = ctx.ratings; m = r > 0
        if m.sum() == 0:
            return np.clip(self.mu + self.bi, 1, 5)
        bu = ((r - self.mu - self.bi) * m).sum() / (m.sum() + 5)
        e = (r - self.mu - self.bi - bu) * m
        Qi = self.Q[m]
        p = np.linalg.solve(Qi.T @ Qi + np.eye(self.k) * self.reg, Qi.T @ e[m])
        return np.clip(self.mu + self.bi + bu + self.Q @ p, 1, 5)

    def scores(self, ctx):
        return self.predict(ctx)


DEFAULT_WEIGHTS = dict(w_mf=1.0, w_cf=0.5, w_cb=0.5, w_pop=0.25, w_profile=1.0, w_diff=0.3, gamma=0.15)


class HybridRecommender(BaseRecommender):
    """STUDENT'S OWN IMPROVEMENT - PCHR (Prerequisite-, Career- and difficulty-aware Hybrid
    Recommender with MMR diversity re-ranking).

    score(j) = w_mf*z(MF) + w_cf*z(kNN) + w_cb*z(Content) + w_pop*z(log popularity)
             + w_profile*profile_fit(career goal, declared interests)
             - w_diff*max(0, difficulty_j - ability(CGPA))
    * hard rule: only courses whose prerequisites are done and whose year-level is open
    * final list built greedily with MMR:  (1-gamma)*score_norm - gamma*max_sim_to_selected
    * cold start: with no ratings the z-terms vanish and profile_fit + popularity take over
    """
    name = "Hybrid PCHR"

    def __init__(self, base=None, **weights):
        self.w = {**DEFAULT_WEIGHTS, **weights}
        self.base = base            # optional shared, already-fitted (mf, knn, cb, pop)

    def fit(self, data, R):
        super().fit(data, R)
        if self.base is None:
            self.mf = MatrixFactorization().fit(data, R)
            self.knn = UserKNNRecommender().fit(data, R)
            self.cb = ContentBasedRecommender().fit(data, R)
        else:
            self.mf, self.knn, self.cb = self.base
        self.S = self.cb.S
        return self

    # ---- components (cacheable per user) --------------------------------------------------
    def components(self, ctx):
        d = self.data
        career = np.array(CAREERS[ctx.career]); career = career / career.max()
        career_fit = career[d.topic]
        int_vec = np.zeros(len(TOPICS))
        for t in ctx.interests:
            if t in TOPIC_IDX:
                int_vec[TOPIC_IDX[t]] = 1.0
        profile = 0.5 * (career_fit + int_vec[d.topic])
        gap = np.maximum(0, d.difficulty - gpa_to_ability(ctx.gpa))
        return dict(mf=zscore(self.mf.predict(ctx)), cf=zscore(self.knn.predict(ctx)),
                    cb=zscore(self.cb.scores(ctx)), pop=zscore(np.log1p(self.counts)),
                    profile=profile, gap=gap)

    def combine(self, comp, w=None):
        w = w or self.w
        return (w["w_mf"] * comp["mf"] + w["w_cf"] * comp["cf"] + w["w_cb"] * comp["cb"] +
                w["w_pop"] * comp["pop"] + w["w_profile"] * comp["profile"] * 2 -
                w["w_diff"] * comp["gap"])

    def scores(self, ctx):
        return self.combine(self.components(ctx))

    def rank(self, ctx, k, allowed, comp=None, w=None):
        w = w or self.w
        comp = comp or self.components(ctx)
        s = self.combine(comp, w)
        allowed = allowed & self.data.eligible(ctx.taken, ctx.next_semester)   # prerequisite rule
        cand = list(np.where(allowed)[0])
        if not cand:
            return []
        lo, hi = s[cand].min(), s[cand].max()
        sn = (s - lo) / (hi - lo + 1e-9)
        chosen = []
        while cand and len(chosen) < k:
            if chosen and w["gamma"] > 0:
                red = self.S[np.ix_(cand, chosen)].max(1)
                mmr = (1 - w["gamma"]) * sn[cand] - w["gamma"] * red
            else:
                mmr = sn[cand]
            best = cand[int(np.argmax(mmr))]
            chosen.append(best); cand.remove(best)
        return chosen

    def explain(self, ctx, j, comp=None):
        d = self.data; comp = comp or self.components(ctx); why = []
        if comp["profile"][j] >= 0.5:
            why.append(f"fits your goal/interests ({ctx.career}; {d.courses.topic[j]})")
        liked = [i for i in np.argsort(-self.S[j]) if ctx.ratings[i] >= 4 and self.S[j, i] > 0.08][:2]
        if liked:
            why.append("similar to courses you rated highly: " + ", ".join(d.courses.name[i] for i in liked))
        if comp["cf"][j] > 0.8:
            why.append("students with similar taste rated it highly")
        pre = [d.item_ids[p] for p in np.where(d.P[j])[0]]
        if pre:
            why.append("prerequisites done: " + ", ".join(pre))
        why.append("difficulty %d/5 suits your CGPA" % d.difficulty[j] if comp["gap"][j] == 0
                   else "challenging for your CGPA (difficulty %d/5)" % d.difficulty[j])
        return why


# ============================================================================================
# 5. EVALUATION: metrics, tuning, ablation, cold start
# ============================================================================================
REL_THRESHOLD = 4       # a held-out course counts as "relevant" if rated >= 4


# ---------------------------------------------------------------- metrics
def ranking_metrics(recs, relevant, k):
    hits = [1 if j in relevant else 0 for j in recs[:k]]
    h = sum(hits)
    dcg = sum(x / np.log2(i + 2) for i, x in enumerate(hits))
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dict(precision=h / k, recall=h / len(relevant), ndcg=dcg / idcg, hit=float(h > 0))


def intra_list_diversity(recs, S):
    if len(recs) < 2:
        return 0.0
    sub = S[np.ix_(recs, recs)]
    n = len(recs)
    return 1.0 - (sub.sum() - np.trace(sub)) / (n * (n - 1))


# ---------------------------------------------------------------- protocol
def build_split(data, drop_last=0):
    train, test, tsem = temporal_split(data, drop_last)
    return train, test, tsem, data.matrix(train)


def evaluate(data, models, train, test, tsem, R, k=5, eligible_for_all=False,
             history_semesters=None, S=None):
    """Return a DataFrame (one row per model). history_semesters=n keeps only the ratings of
    the first n semesters (transcript still known) -> cold-start experiments."""
    rel = test[test.rating >= REL_THRESHOLD].groupby("student_id").course_id.apply(
        lambda s: {data.item_idx[c] for c in s}).to_dict()
    test_by_u = {u: g for u, g in test.groupby("student_id")}
    tr_sem = train.merge(data.students[["student_id"]], on="student_id")
    if history_semesters is not None:
        R_hist = data.matrix(train[train.semester <= history_semesters])
    users = [u for u in rel if len(rel[u]) > 0]
    rows = []
    for m in models:
        acc, recs_all, elig_rate, se, ae, n_r = [], [], [], 0.0, 0.0, 0
        for u in users:
            ui = data.user_idx[u]
            ctx = make_context(data, R, u, next_semester=tsem[u])
            if history_semesters is not None:
                ctx.ratings = R_hist[ui]
            elig = data.eligible(ctx.taken, ctx.next_semester)
            allowed = elig if eligible_for_all else ~ctx.taken
            recs = m.rank(ctx, k, allowed)
            acc.append(ranking_metrics(recs, rel[u], k))
            recs_all.append(recs)
            elig_rate.append(np.mean([elig[j] for j in recs]) if recs else 0)
            if history_semesters is None:               # rating-prediction error
                pred = m.predict(ctx)
                if pred is not None:
                    g = test_by_u[u]
                    p = pred[g.course_id.map(data.item_idx).values]
                    se += ((p - g.rating.values) ** 2).sum(); ae += np.abs(p - g.rating.values).sum()
                    n_r += len(g)
        row = pd.DataFrame(acc).mean().to_dict()
        row.update(model=m.name, coverage=len({j for r in recs_all for j in r}) / data.n_items,
                   diversity=np.mean([intra_list_diversity(r, S) for r in recs_all]),
                   topic_spread=np.mean([len({data.topic[j] for j in r}) / max(len(r), 1) for r in recs_all]),
                   eligible_rate=np.mean(elig_rate),
                   rmse=np.sqrt(se / n_r) if n_r else np.nan, mae=ae / n_r if n_r else np.nan)
        rows.append(row)
    cols = ["model", "precision", "recall", "ndcg", "hit", "coverage", "diversity", "topic_spread",
            "eligible_rate", "rmse", "mae"]
    return pd.DataFrame(rows)[cols]


def fit_all(data, R):
    pop = PopularityRecommender().fit(data, R)
    cb = ContentBasedRecommender().fit(data, R)
    knn = UserKNNRecommender().fit(data, R)
    mf = MatrixFactorization().fit(data, R)
    return pop, cb, knn, mf


# ---------------------------------------------------------------- tuning on a validation split
def tune_hybrid(data, k=5, verbose=False):
    """Grid-search hybrid weights on the semester BEFORE the test semester (no test leakage)."""
    train, val, tsem, R = build_split(data, drop_last=1)
    pop, cb, knn, mf = fit_all(data, R)
    hyb = HybridRecommender(base=(mf, knn, cb)).fit(data, R)
    rel = val[val.rating >= REL_THRESHOLD].groupby("student_id").course_id.apply(
        lambda s: {data.item_idx[c] for c in s}).to_dict()
    cache = []
    for u in rel:
        ctx = make_context(data, R, u, next_semester=tsem[u])
        cache.append((ctx, hyb.components(ctx), rel[u]))
    grid = dict(w_mf=[0.5, 1.0], w_cf=[0, 0.5], w_cb=[0, 0.5], w_pop=[0, 0.25],
                w_profile=[0, 0.5, 1.0], w_diff=[0, 0.3])
    best, best_score = None, -1
    for vals in itertools.product(*grid.values()):
        w = {**DEFAULT_WEIGHTS, **dict(zip(grid, vals)), "gamma": 0.0}
        if w["w_mf"] + w["w_cf"] + w["w_cb"] == 0:
            continue
        sc = np.mean([ranking_metrics(hyb.rank(c, k, ~c.taken, comp=cm, w=w), r, k)["ndcg"]
                      for c, cm, r in cache])
        if sc > best_score:
            best, best_score = w, sc
    best["gamma"] = DEFAULT_WEIGHTS["gamma"]
    if verbose:
        print("tuned weights:", best, "val NDCG=%.3f" % best_score)
    return best


# ---------------------------------------------------------------- full experiment for one seed
def run_seed(seed, n_students=600, k=5, verbose=False):
    courses, students, ratings = generate_dataset(n_students, seed)
    data = CourseData(courses, students, ratings)
    w = tune_hybrid(data, k, verbose)
    train, test, tsem, R = build_split(data)
    pop, cb, knn, mf = fit_all(data, R)
    hyb = HybridRecommender(base=(mf, knn, cb), **w).fit(data, R)
    models = [RandomRecommender().fit(data, R), pop, cb, knn, mf, hyb]
    out = {"weights": w}
    out["main"] = evaluate(data, models, train, test, tsem, R, k, False, S=cb.S)
    out["fair"] = evaluate(data, models, train, test, tsem, R, k, True, S=cb.S)
    out["main10"] = evaluate(data, models, train, test, tsem, R, 10, False, S=cb.S)

    # ablation: remove one component of the hybrid at a time
    abl = {"Full PCHR": {}, "- eligibility rule": "noelig", "- career/interest profile": dict(w_profile=0),
           "- difficulty-vs-CGPA penalty": dict(w_diff=0), "- MMR diversity": dict(gamma=0),
           "- matrix factorization": dict(w_mf=0, w_cf=0.5, w_cb=0.5)}
    rows = []
    for name, ch in abl.items():
        if ch == "noelig":
            h = HybridRecommender(base=(mf, knn, cb), **w).fit(data, R)
            h.rank = lambda ctx, k_, allowed, _h=h: BaseHybridRank(_h, ctx, k_, allowed)
        else:
            h = HybridRecommender(base=(mf, knn, cb), **{**w, **ch}).fit(data, R)
        h.name = name
        rows.append(h)
    out["ablation"] = evaluate(data, rows, train, test, tsem, R, k, False, S=cb.S)

    # diversity-accuracy trade-off (gamma sweep)
    sweep = []
    for g in [0, 0.1, 0.2, 0.3, 0.5]:
        h = HybridRecommender(base=(mf, knn, cb), **{**w, "gamma": g}).fit(data, R); h.name = f"gamma={g}"
        sweep.append(h)
    out["gamma"] = evaluate(data, sweep, train, test, tsem, R, k, False, S=cb.S)

    # cold-start: number of semesters of ratings available (transcript always known)
    cs = []
    for n in [0, 1, 2, 3]:
        r = evaluate(data, models, train, test, tsem, R, k, True, history_semesters=n, S=cb.S)
        r["history_semesters"] = n; cs.append(r)
    out["cold"] = pd.concat(cs)
    return out


def BaseHybridRank(h, ctx, k, allowed):
    """Hybrid ranking WITHOUT the prerequisite rule (used only for the ablation)."""
    comp = h.components(ctx); s = h.combine(comp)
    s = np.where(allowed, s, -np.inf)
    return [int(j) for j in np.argsort(-s)[:k] if np.isfinite(s[j])]


# ============================================================================================
# 6. COMMAND-LINE INTERFACE: compare and demo
# ============================================================================================
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
RES = "results"


# ------------------------------------------------------------------ compare
def cmd_compare(a):
    os.makedirs(RES, exist_ok=True)
    seeds = list(range(a.seeds))
    runs = [run_seed(s, a.students, a.k) for s in seeds]
    print(f"\n=== {a.seeds} random datasets x {a.students} students, K={a.k}, "
          f"relevant = held-out course rated >= 4 ===")

    def agg(key, idx="model"):
        df = pd.concat(r[key] for r in runs)
        g = df.groupby(idx, sort=False)
        return g.mean(numeric_only=True), g.std(numeric_only=True)

    def show(title, key, cols, fname):
        m, s = agg(key)
        t = m[cols].round(3).astype(str) + " +/-" + s[cols].round(3).astype(str)
        t = t.where(m[cols].notna(), "n/a")
        print(f"\n--- {title} ---"); print(t.to_string())
        m.round(4).to_csv(f"{RES}/{fname}.csv")
        return m, s

    cols = ["precision", "recall", "ndcg", "hit", "coverage", "topic_spread", "eligible_rate"]
    m1, s1 = show("TABLE 1: all courses not yet taken are candidates", "main", cols, "table1_raw")
    m2, s2 = show("TABLE 2 (fair): every model may only recommend ELIGIBLE courses", "fair", cols, "table2_fair")
    show("TABLE 3: rating-prediction error on held-out ratings (lower is better)", "main", ["rmse", "mae"], "table3_rmse")
    show("TABLE 4: ablation of the student-designed improvement", "ablation",
         ["precision", "recall", "ndcg", "topic_spread", "eligible_rate"], "table4_ablation")
    show("TABLE 5: diversity vs accuracy (MMR gamma)", "gamma", ["precision", "ndcg", "topic_spread"], "table5_gamma")
    cold = pd.concat(r["cold"] for r in runs)
    piv = cold.groupby(["history_semesters", "model"], sort=False).ndcg.mean().unstack()
    piv = piv[[c for c in pd.concat(r["main"] for r in runs).model.unique()]]
    print("\n--- TABLE 6: NDCG@K vs number of semesters of ratings known (cold start) ---")
    print(piv.round(3).to_string()); piv.round(4).to_csv(f"{RES}/table6_coldstart.csv")
    print("\ntuned hybrid weights (seed 0):", runs[0]["weights"])

    # significance: paired difference in NDCG across seeds, hybrid vs best baseline (fair table)
    hyb_name = "Hybrid PCHR"
    base = [n for n in m2.index if n not in (hyb_name, "Random")]   # works even if Random is absent
    best = m2.loc[base, "ndcg"].idxmax()
    diffs = [r["fair"].set_index("model").ndcg[hyb_name] - r["fair"].set_index("model").ndcg[best] for r in runs]
    print(f"\nHybrid - {best} (fair NDCG): mean {np.mean(diffs):+.3f}, min {np.min(diffs):+.3f}, "
          f"wins in {sum(d > 0 for d in diffs)}/{len(diffs)} datasets")
    make_plots(m2, s2, runs, piv)
    print(f"\nSaved tables and plots in ./{RES}/")


def make_plots(m2, s2, runs, piv):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    names = list(m2.index); colors = ["#bbb", "#9bb", "#79b", "#58a", "#36a", "#d55"]
    ax[0].bar(range(len(names)), m2.ndcg, yerr=s2.ndcg, color=colors[:len(names)], capsize=3)
    ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=7, rotation=25, ha="right")
    ax[0].set_title("NDCG@K (eligible candidates, mean+/-std)"); ax[0].set_ylabel("NDCG")
    abl = pd.concat(r["ablation"] for r in runs).groupby("model", sort=False).ndcg.mean()
    ax[1].barh(range(len(abl)), abl.values, color="#d55"); ax[1].set_yticks(range(len(abl)))
    ax[1].set_yticklabels(abl.index, fontsize=8); ax[1].invert_yaxis(); ax[1].set_title("Ablation of hybrid (NDCG@K)")
    for c in piv.columns:
        ax[2].plot(piv.index, piv[c], marker="o", label=c, lw=3 if "Hybrid" in c else 1.3)
    ax[2].set_xlabel("semesters of ratings known"); ax[2].set_ylabel("NDCG"); ax[2].legend(fontsize=7, loc="lower right")
    ax[2].set_title("Cold-start behaviour")
    plt.tight_layout(); plt.savefig(f"{RES}/comparison.png", dpi=130)


# ------------------------------------------------------------------ demo
def load_or_make(seed=42):
    if not os.path.exists("data/ratings.csv"):
        save_dataset(*generate_dataset(600, seed))
    return CourseData(pd.read_csv("data/courses.csv"), pd.read_csv("data/students.csv"),
                      pd.read_csv("data/ratings.csv"))


def print_recs(name, data, model, ctx, k, allowed, show_why=False):
    recs = model.rank(ctx, k, allowed)
    print(f"\n{name}")
    comp = model.components(ctx) if show_why else None
    for r, j in enumerate(recs, 1):
        c = data.courses.iloc[j]
        print(f"  {r}. {c.course_id}  {c['name']:<38} [{c.topic}, year {c.level}, difficulty {c.difficulty}]")
        if show_why:
            for w in model.explain(ctx, j, comp):
                print(f"       - {w}")


def cmd_demo(a):
    data = load_or_make()
    print("Tuning hybrid weights on validation semester and training models ...")
    w = tune_hybrid(data, a.k)
    train, test, tsem, R = build_split(data)
    pop, cb, knn, mf = fit_all(data, R)
    hyb = HybridRecommender(base=(mf, knn, cb), **w).fit(data, R)

    if a.new:                                            # ---- cold-start student
        print("\n== New student ==")
        career = choose("Career goal", list(CAREERS))
        ints = choose("Interests (choose 2)", TOPICS, 2)
        gpa = float(input("CGPA (5-10): ") or 7.5)
        sem = int(input("Semester you are entering (1-8): ") or 5)
        done = input("Completed course IDs, comma separated (e.g. CS101,CS102,CS201): ")
        taken = np.zeros(data.n_items, bool)
        rat = np.zeros(data.n_items)
        for cid in [x.strip().upper() for x in done.split(",") if x.strip()]:
            if cid in data.item_idx:
                taken[data.item_idx[cid]] = True
                rat[data.item_idx[cid]] = float(input(f"  rating for {cid} (1-5, 0=skip): ") or 0)
        ctx = UserContext(rat, taken, gpa, career, ints, sem, -1)
        print_recs("Popularity baseline (ignores you)", data, pop, ctx, a.k, data.eligible(taken, sem))
        print_recs("HYBRID PCHR (personalised)", data, hyb, ctx, a.k, ~taken, show_why=True)
        return

    sid = a.student
    if sid is None:
        sid = data.user_ids[int(input(f"Student index 0-{data.n_users-1}: ") or 0)]
    ctx = make_context(data, R, sid, next_semester=tsem[sid])
    row = data.students[data.students.student_id == sid].iloc[0]
    print(f"\n== {sid}: {row.branch}, CGPA {row.gpa}, goal '{row.career_goal}', "
          f"interests {row.declared_interests}, entering semester {ctx.next_semester} ==")
    print("Completed & rated:", ", ".join(f"{data.item_ids[j]}({int(ctx.ratings[j])})"
                                          for j in np.where(ctx.taken)[0]))
    el = data.eligible(ctx.taken, ctx.next_semester)
    for m in (pop, cb, knn, mf):
        print_recs(m.name, data, m, ctx, a.k, ~ctx.taken)
    print_recs("HYBRID PCHR - with explanations", data, hyb, ctx, a.k, ~ctx.taken, show_why=True)
    actual = test[(test.student_id == sid)]
    print("\nWhat the student actually took next semester (ground truth):",
          ", ".join(f"{r.course_id}(rated {r.rating})" for r in actual.itertuples()))


def choose(label, options, n=1):
    print(f"{label}:")
    for i, o in enumerate(options, 1):
        print(f"  {i}. {o}")
    picks = [options[int(x) - 1] for x in input(f"  number{'s (comma separated)' if n > 1 else ''}: ").split(",")]
    return picks[0] if n == 1 else picks[:n]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("compare"); c.add_argument("--seeds", type=int, default=5)
    c.add_argument("--students", type=int, default=600); c.add_argument("--k", type=int, default=5)
    c.set_defaults(fn=cmd_compare)
    d = sp.add_parser("demo"); d.add_argument("--student"); d.add_argument("--new", action="store_true")
    d.add_argument("--k", type=int, default=5); d.set_defaults(fn=cmd_demo)
    g = sp.add_parser("generate"); g.add_argument("--seed", type=int, default=42)
    g.set_defaults(fn=lambda a: (save_dataset(*generate_dataset(600, a.seed)), print("saved to ./data")))
    a = ap.parse_args(); a.fn(a)
