"""Personalized Course Recommendation System - command line interface.

  python main.py compare              # multi-seed comparison of all algorithms (tables + plots)
  python main.py demo                 # interactive demo: pick an existing student
  python main.py demo --student S0007 # recommendations + explanations for one student
  python main.py demo --new           # cold-start demo: type your own profile
"""
import argparse, os, json
import numpy as np
import pandas as pd

from recsys.data_generator import generate_dataset, save_dataset
from recsys.dataset import CourseData, make_context, UserContext, temporal_split
from recsys.catalog import TOPICS, CAREERS
from recsys.evaluate import run_seed, tune_hybrid, fit_all, build_split
from recsys.models import HybridRecommender, RandomRecommender

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
    best = m2.drop(index=["Hybrid PCHR (ours)", "Random"]).ndcg.idxmax()
    diffs = [r["fair"].set_index("model").ndcg["Hybrid PCHR (ours)"] - r["fair"].set_index("model").ndcg[best] for r in runs]
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
    ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=7)
    ax[0].set_title("NDCG@K (eligible candidates, mean+/-std)"); ax[0].set_ylabel("NDCG")
    abl = pd.concat(r["ablation"] for r in runs).groupby("model", sort=False).ndcg.mean()
    ax[1].barh(range(len(abl)), abl.values, color="#d55"); ax[1].set_yticks(range(len(abl)))
    ax[1].set_yticklabels(abl.index, fontsize=8); ax[1].invert_yaxis(); ax[1].set_title("Ablation of hybrid (NDCG@K)")
    for c in piv.columns:
        ax[2].plot(piv.index, piv[c], marker="o", label=c, lw=3 if "ours" in c else 1.3)
    ax[2].set_xlabel("semesters of ratings known"); ax[2].set_ylabel("NDCG"); ax[2].legend(fontsize=7)
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
    print_recs("HYBRID PCHR (ours) - with explanations", data, hyb, ctx, a.k, ~ctx.taken, show_why=True)
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
