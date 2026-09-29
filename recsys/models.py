"""Recommendation algorithms.

Baseline : PopularityRecommender
Approach 1: ContentBasedRecommender   (TF-IDF + cosine similarity)
Approach 2: UserKNNRecommender        (user-based collaborative filtering)
Approach 3: MatrixFactorization       (biased ALS, from scratch in numpy)
Student's improvement: HybridRecommender (see class docstring)
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .catalog import TOPICS, TOPIC_IDX, CAREERS
from .data_generator import gpa_to_ability


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
    name = "Hybrid PCHR (ours)"

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
