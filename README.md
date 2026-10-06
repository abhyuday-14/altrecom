# Personalized Course Recommendation System

Compares **five AI approaches** for recommending elective courses to a student, adds a
**student-designed improvement (PCHR)**, reports **quantitative results** and ends with a
**working command-line demo**.

```
pip install -r requirements.txt
python main.py generate          # (optional) create data/*.csv
python main.py compare           # tables + results/comparison.png  (~1 min)
python main.py demo --student S0007   # recommendations from every model + explanations
python main.py demo --new             # cold-start: enter your own profile
```

## Project layout
| File | Purpose |
|---|---|
| `recsys/catalog.py` | 48-course catalogue (topic, year, difficulty, prerequisites), career goals |
| `recsys/data_generator.py` | Synthetic students/ratings with documented behavioural rules (CSV output) |
| `recsys/dataset.py` | Data container, eligibility rule, user context, temporal split |
| `recsys/models.py` | Random, Popularity, Content-Based, User-kNN CF, Matrix Factorization (ALS), **Hybrid PCHR** |
| `recsys/evaluate.py` | Metrics, tuning, ablation, cold-start experiment |
| `main.py` | `compare` and `demo` commands |

## Functionalities
| Component | Where it is covered |
|---|---|
| Problem formulation & originality | Top-N course recommendation with prerequisites, year-gating, CGPA/difficulty and career goal – constraints ordinary movie-style recommenders ignore |
| AI concepts / algorithm implementation | TF-IDF + cosine, user-kNN with mean-centring & shrinkage, ALS matrix factorisation written from scratch in numpy, hybrid scoring, MMR re-ranking |
| Comparison of algorithms | 5 models + Random floor; Tables 1-3 in `compare` |
| Dataset / environment / experimentation | Generator, leave-last-semester-out split, validation-split tuning, 5 random datasets (mean ± std), cold-start study |
| Evaluation & interpretation | Precision/Recall/NDCG/HitRate@K, RMSE/MAE, coverage, topic spread, eligibility rate, ablation |
| Student's own improvement | **PCHR** (see `HybridRecommender` docstring) – ablation Table 4 shows which parts matter |
| Working application / demo | `python main.py demo` (existing student, explanations, cold-start mode) |

## Headline results (K = 5, mean of 5 datasets; every model restricted to eligible courses)
| Model | Precision | Recall | NDCG |
|---|---|---|---|
| Random | 0.208 | 0.544 | 0.382 |
| Popularity | 0.153 | 0.417 | 0.275 |
| Content-Based | 0.283 | 0.716 | 0.565 |
| User-kNN CF | 0.325 | 0.831 | 0.713 |
| Matrix Factorization | 0.331 | 0.849 | 0.708 |
| **Hybrid PCHR** | **0.366** | **0.924** | **0.805** |

Hybrid beat the strongest baseline in 5/5 datasets. Regenerate exact numbers with `python main.py compare`.

## Limitations 
* Data is **synthetic**: results show the methods work under the generator's stated rules, not on real students. The CSV schema lets you drop in real data.
* The eligible pool is small (~9 courses), so Random is a strong floor (NDCG 0.38) – read gains relative to it.
* Difficulty penalty and the extra CF/content terms add ~0 in the ablation on this data; the eligibility rule and career/interest profile give the gains.
