"""Metrics, experiment protocol, hyper-parameter tuning, ablation and cold-start study."""
import itertools
import numpy as np
import pandas as pd
from .dataset import CourseData, make_context, temporal_split
from .models import (PopularityRecommender, ContentBasedRecommender, UserKNNRecommender,
                     MatrixFactorization, HybridRecommender, RandomRecommender, DEFAULT_WEIGHTS)
from .data_generator import generate_dataset

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
