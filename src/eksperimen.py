"""
Pengaruh penanganan missing values terhadap akurasi KNN + Grid Search
pada dataset diabetes (Pima).

Menjalankan seluruh eksperimen pada laporan:
  - 8 skenario penanganan missing value + 1 skenario bocor (tidak valid)
  - nested cross-validation 10-fold x 3 ulangan (inner 5-fold Grid Search)
  - hold-out 80:20 (random_state 42)
  - corrected resampled t-test (Nadeau & Bengio, 2003)
  - galat imputasi (RMSE) pada nilai yang sengaja disembunyikan

Hasil ditulis ke folder hasil/ (JSON, CSV, PNG, HTML).
"""
import argparse
import json
import os
import shutil
import tempfile
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from joblib import Memory
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
VAR4 = ["Glucose", "BloodPressure", "Insulin", "BMI"]
VAR5 = VAR4 + ["SkinThickness"]
GRID = {
    "knn__n_neighbors": list(range(1, 50, 2)),
    "knn__weights": ["uniform", "distance"],
    "knn__p": [1, 2],
}


def log(msg, progress=None):
    print(msg, flush=True)
    if progress:
        progress(msg)


# ----------------------------------------------------------------- data
def load(path):
    df = pd.read_csv(path)
    X = df.drop(columns="Outcome")
    y = df["Outcome"].to_numpy()
    return df, X, y


def zero_to_nan(X, cols):
    X = X.copy()
    X[cols] = X[cols].replace(0, np.nan)
    return X


# ------------------------------------------------------------ pipelines
def make_pipe(imputer, memory):
    knn = KNeighborsClassifier()
    if imputer is None:
        return Pipeline([("sc", StandardScaler()), ("knn", knn)])
    return Pipeline(
        [("sc1", StandardScaler()), ("imp", imputer),
         ("sc2", StandardScaler()), ("knn", knn)],
        memory=memory,
    )


def scenarios(X, y):
    """name -> (X, y, imputer factory, comparable_with_baseline)"""
    X4 = zero_to_nan(X, VAR4)
    X5 = zero_to_nan(X, VAR5)
    complete = ~(X[VAR4] == 0).any(axis=1).to_numpy()
    mice = lambda: IterativeImputer(max_iter=25, random_state=SEED)  # noqa: E731
    return {
        "Tanpa penanganan": (X, y, lambda: None, True),
        "Hapus baris": (X[complete].reset_index(drop=True), y[complete], lambda: None, False),
        "Mean": (X4, y, lambda: SimpleImputer(strategy="mean"), True),
        "Median": (X4, y, lambda: SimpleImputer(strategy="median"), True),
        "Median + indikator": (X4, y, lambda: SimpleImputer(strategy="median", add_indicator=True), True),
        "KNN Imputer": (X4, y, lambda: KNNImputer(n_neighbors=5, weights="distance"), True),
        "MICE": (X4, y, mice, True),
        "MICE + SkinThickness": (X5, y, mice, True),
    }


# ------------------------------------------------------------ evaluasi
def nested_cv(Xs, ys, imp_factory, outer, inner, memory, n_jobs):
    rows = []
    for i, (tr, te) in enumerate(outer.split(Xs, ys)):
        Xtr, Xte, ytr, yte = Xs.iloc[tr], Xs.iloc[te], ys[tr], ys[te]
        gs = GridSearchCV(make_pipe(imp_factory(), memory), GRID, cv=inner,
                          scoring="accuracy", n_jobs=n_jobs)
        gs.fit(Xtr, ytr)
        pred = gs.predict(Xte)
        proba = gs.predict_proba(Xte)[:, 1]
        base = make_pipe(imp_factory(), None).set_params(knn__n_neighbors=5).fit(Xtr, ytr)
        bp = gs.best_params_
        rows.append({
            "fold": i,
            "acc": accuracy_score(yte, pred),
            "f1": f1_score(yte, pred),
            "recall": recall_score(yte, pred),
            "auc": roc_auc_score(yte, proba),
            "acc_k5": accuracy_score(yte, base.predict(Xte)),
            "k": bp["knn__n_neighbors"], "weights": bp["knn__weights"], "p": bp["knn__p"],
        })
    return pd.DataFrame(rows)


def holdout(Xs, ys, imp_factory, inner, memory, n_jobs):
    Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=0.2, stratify=ys, random_state=SEED)
    gs = GridSearchCV(make_pipe(imp_factory(), memory), GRID, cv=inner,
                      scoring="accuracy", n_jobs=n_jobs).fit(Xtr, ytr)
    bp = gs.best_params_
    return {
        "acc": accuracy_score(yte, gs.predict(Xte)),
        "params": f"k={bp['knn__n_neighbors']}, {bp['knn__weights']}, p={bp['knn__p']}",
    }


def corrected_ttest(a, b, test_frac):
    """Corrected resampled t-test (Nadeau & Bengio, 2003); returns two-sided p."""
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 1.0
    t = d.mean() / np.sqrt((1 / k + test_frac / (1 - test_frac)) * var)
    return float(2 * stats.t.sf(abs(t), df=k - 1))


def corrected_ci(acc, test_frac, level=0.95):
    k = len(acc)
    se = np.sqrt((1 / k + test_frac / (1 - test_frac)) * np.var(acc, ddof=1))
    h = stats.t.ppf(0.5 + level / 2, df=k - 1) * se
    return float(np.mean(acc) - h), float(np.mean(acc) + h)


def imputation_rmse(X, reps, frac=0.2):
    complete = ~(X[VAR4] == 0).any(axis=1).to_numpy()
    Xc = X[complete].to_numpy(dtype=float)
    Z = (Xc - Xc.mean(0)) / Xc.std(0)
    cols = [X.columns.get_loc(c) for c in VAR4]
    imps = {
        "MICE": lambda: IterativeImputer(max_iter=25, random_state=SEED),
        "KNN Imputer": lambda: KNNImputer(n_neighbors=5, weights="distance"),
        "Mean": lambda: SimpleImputer(strategy="mean"),
        "Median": lambda: SimpleImputer(strategy="median"),
    }
    out = {m: [] for m in imps}
    for r in range(reps):
        rng = np.random.RandomState(r)
        mask = np.zeros_like(Z, dtype=bool)
        mask[:, cols] = rng.rand(Z.shape[0], len(cols)) < frac
        Zm = Z.copy()
        Zm[mask] = np.nan
        for m, f in imps.items():
            Zi = f().fit_transform(Zm)
            out[m].append(float(np.sqrt(np.mean((Zi[mask] - Z[mask]) ** 2))))
    return {m: (float(np.mean(v)), float(np.std(v, ddof=1))) for m, v in out.items()}, int(complete.sum())


def leaky_median_per_class(X, y):
    """TIDAK VALID: median per kelas Outcome dihitung dari seluruh data (termasuk label data uji)."""
    X4 = zero_to_nan(X, VAR4)
    for c in VAR4:
        for cls in (0, 1):
            m = X4.loc[y == cls, c].median()
            X4.loc[(y == cls) & X4[c].isna(), c] = m
    return X4


# --------------------------------------------------------------- utama
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/diabetes.csv")
    ap.add_argument("--out", default="hasil")
    ap.add_argument("--quick", action="store_true", help="uji cepat: 3-fold x 1 ulangan, RMSE 3 ulangan")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    progress_file = os.path.join(args.out, "progres.txt")

    def progress(msg):
        with open(progress_file, "a", encoding="utf-8") as f:
            f.write(time.strftime("%H:%M:%S ") + msg + "\n")

    open(progress_file, "w").close()
    t0 = time.time()

    df, X, y = load(args.data)
    n_split, n_rep = (3, 1) if args.quick else (10, 3)
    outer = RepeatedStratifiedKFold(n_splits=n_split, n_repeats=n_rep, random_state=SEED)
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    test_frac = 1 / n_split
    cache = tempfile.mkdtemp()
    memory = Memory(cache, verbose=0)

    info = {
        "baris": int(len(df)), "positif": int(y.sum()),
        "persen_positif": float(y.mean() * 100),
        "nol_per_variabel": {c: int((X[c] == 0).sum()) for c in VAR5},
        "baris_lengkap_4_variabel": int((~(X[VAR4] == 0).any(axis=1)).sum()),
        "outer": f"{n_split}-fold x {n_rep} ulangan", "inner": "5-fold stratified",
        "grid_kombinasi": int(np.prod([len(v) for v in GRID.values()])),
    }
    log(f"Data: {info['baris']} baris, {info['positif']} positif. Nested CV {info['outer']}.", progress)

    sc = scenarios(X, y)
    folds, hold = {}, {}
    for name, (Xs, ys, f, _) in sc.items():
        t = time.time()
        log(f"[{name}] nested CV ...", progress)
        folds[name] = nested_cv(Xs, ys, f, outer, inner, memory, args.jobs)
        hold[name] = holdout(Xs, ys, f, inner, memory, args.jobs)
        log(f"[{name}] akurasi {folds[name]['acc'].mean()*100:.2f}% "
            f"(hold-out {hold[name]['acc']*100:.2f}%), {time.time()-t:.0f} dtk", progress)

    log("[Median per kelas - TIDAK VALID] nested CV ...", progress)
    leak = nested_cv(leaky_median_per_class(X, y), y, lambda: None, outer, inner, memory, args.jobs)

    log("Galat imputasi (RMSE) ...", progress)
    rmse, n_complete = imputation_rmse(X, reps=3 if args.quick else 30)
    shutil.rmtree(cache, ignore_errors=True)

    # ---- ringkas
    base = folds["Tanpa penanganan"]
    tab3, tab4 = [], []
    for name, d in folds.items():
        comparable = sc[name][3] and name != "Tanpa penanganan"
        diff = (d["acc"].mean() - base["acc"].mean()) * 100
        tab3.append({
            "Skenario": name,
            "Akurasi nested CV (%)": round(d["acc"].mean() * 100, 2),
            "SD (pp)": round(d["acc"].std(ddof=1) * 100, 2),
            "Selisih (pp)": round(diff, 2) if comparable else None,
            "p": round(corrected_ttest(d["acc"], base["acc"], test_frac), 3) if comparable else None,
            "F1": round(d["f1"].mean(), 3),
            "Recall": round(d["recall"].mean(), 3),
            "AUC": round(d["auc"].mean(), 3),
            "Akurasi hold-out (%)": round(hold[name]["acc"] * 100, 2),
        })
        w = Counter(d["weights"]).most_common(1)[0]
        tab4.append({
            "Skenario": name,
            "KNN k=5 tanpa grid (%)": round(d["acc_k5"].mean() * 100, 2),
            "Dengan grid (%)": round(d["acc"].mean() * 100, 2),
            "Kenaikan dari grid (pp)": round((d["acc"].mean() - d["acc_k5"].mean()) * 100, 2),
            "Parameter terbaik (hold-out)": hold[name]["params"],
            "Median k (nested)": float(np.median(d["k"])),
            "Rentang k (nested)": f"{d['k'].min()}-{d['k'].max()}",
            "Weights terbanyak": f"{w[0]} ({w[1]}/{len(d)})",
        })
    tab3 = pd.DataFrame(tab3)
    tab4 = pd.DataFrame(tab4)
    tab5 = pd.DataFrame([{"Metode": m, "RMSE rata-rata": round(v[0], 3), "SD": round(v[1], 3)}
                         for m, v in sorted(rmse.items(), key=lambda kv: kv[1][0])])
    recall_p = {n: round(corrected_ttest(folds[n]["recall"], base["recall"], test_frac), 3)
                for n in ("Mean", "Median", "KNN Imputer", "MICE")}
    leak_acc = leak["acc"].mean() * 100

    tab3.to_csv(os.path.join(args.out, "tabel_performa.csv"), index=False)
    tab4.to_csv(os.path.join(args.out, "tabel_grid_search.csv"), index=False)
    tab5.to_csv(os.path.join(args.out, "tabel_rmse_imputasi.csv"), index=False)
    pd.concat({k: v for k, v in folds.items()}, names=["skenario"]).to_csv(
        os.path.join(args.out, "akurasi_per_lipatan.csv"))

    ci = {n: corrected_ci(d["acc"].to_numpy() * 100, test_frac)
          for n, d in folds.items() if n != "Hapus baris"}
    ci["Median per kelas (tidak valid)"] = corrected_ci(leak["acc"].to_numpy() * 100, test_frac)
    means = {n: folds[n]["acc"].mean() * 100 for n in ci if n in folds}
    means["Median per kelas (tidak valid)"] = leak_acc
    plot(means, ci, os.path.join(args.out, "grafik_akurasi.png"))

    result = {
        "info": info, "waktu_detik": round(time.time() - t0),
        "tabel_performa": tab3.astype(object).where(tab3.notna(), None).to_dict(orient="records"),
        "tabel_grid_search": tab4.to_dict(orient="records"),
        "tabel_rmse_imputasi": tab5.to_dict(orient="records"),
        "baris_lengkap_rmse": n_complete,
        "p_recall_vs_tanpa": recall_p,
        "median_per_kelas_tidak_valid": {"akurasi": round(leak_acc, 2),
                                         "selisih_pp": round(leak_acc - base["acc"].mean() * 100, 2)},
    }
    with open(os.path.join(args.out, "hasil.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    from laporan_html import write_html  # noqa: E402
    write_html(result, args.out)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n=== Performa KNN + Grid Search ===\n", tab3.to_string(index=False))
    print("\n=== Grid Search ===\n", tab4.to_string(index=False))
    print("\n=== RMSE imputasi ===\n", tab5.to_string(index=False))
    print(f"\nMedian per kelas (TIDAK VALID): {leak_acc:.2f}%")
    log(f"Selesai dalam {time.time()-t0:.0f} detik.", progress)


def plot(means, ci, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(ci.keys())
    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=160)
    base = means["Tanpa penanganan"]
    for i, n in enumerate(names):
        lo, hi = ci[n]
        bad = "tidak valid" in n
        color = "#c0461b" if bad else ("#6b7080" if n == "Tanpa penanganan" else "#2563a8")
        ax.plot([lo, hi], [i, i], color=color, lw=2.2)
        ax.plot(means[n], i, "o", ms=9, color=color, mfc="white" if bad else color, mew=2)
        if n in ("Tanpa penanganan", "MICE") or bad:
            ax.text(means[n], i - 0.28, f"{means[n]:.2f}%".replace(".", ","),
                    ha="center", fontsize=9, color="#333")
    ax.axvline(base, ls="--", color="#9a9ab0", lw=1.2)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlabel("Akurasi nested CV (%), garis = selang kepercayaan 95%")
    ax.grid(axis="x", color="#e5e5e5")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
