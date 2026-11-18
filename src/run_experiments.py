"""Run the WDBC ensemble-learning coursework experiments reproducibly.

Usage (from repository root):
    python -m src.run_experiments
    python -m src.run_experiments --help

No teacher data or locked test labels are accessed. The supplied PDF report
is never modified. All default parameters match the original coursework run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("lesson9")


@dataclass(frozen=True)
class Config:
    seed: int = 1010
    n_estimators: int = 100
    bootstrap_repeats: int = 1000
    stability_repeats: int = 30
    permutation_repeats: int = 10
    gb_max_estimators: int = 500
    jobs: int = 1


def build_models(config: Config) -> dict[str, object]:
    """The seven models used in the original report; no hyperparameter tuning."""
    s, n = config.seed, config.n_estimators
    return {
        "Single tree": DecisionTreeClassifier(max_depth=4, random_state=s),
        "Bagging": BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=s),
            n_estimators=n, max_features=1.0, random_state=s, n_jobs=config.jobs,
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=n, max_features="sqrt", random_state=s, n_jobs=config.jobs,
        ),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=s),
        "Gradient boosting": GradientBoostingClassifier(n_estimators=100, random_state=s),
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000, random_state=s)
        ),
        "RBF-SVM": make_pipeline(
            StandardScaler(), SVC(C=1, kernel="rbf", random_state=s)
        ),
    }


def positive_scores(model: object, data: pd.DataFrame) -> np.ndarray:
    """Return a ranking score: probability where available, otherwise margin."""
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(data))
    return np.asarray(model.predict_proba(data))[:, 1]


def read_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Read only student training and validation; reject invalid/leaky inputs."""
    required = {"sample_id", "malignant"}
    data = []
    for name in ("train.csv", "validation.csv"):
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing student dataset: {path}")
        frame = pd.read_csv(path)
        if not required.issubset(frame.columns):
            raise ValueError(f"{name}: expected columns {sorted(required)}")
        if frame.columns.duplicated().any():
            raise ValueError(f"{name}: duplicate columns")
        if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
            raise ValueError(f"{name}: sample_id must be unique and nonmissing")
        if set(frame["malignant"].dropna().unique()) != {0, 1} or frame["malignant"].isna().any():
            raise ValueError(f"{name}: malignant labels must be binary 0/1")
        data.append(frame)
    train, valid = data
    if train.columns.tolist() != valid.columns.tolist():
        raise ValueError("Training and validation columns/order differ")
    if set(train.sample_id).intersection(set(valid.sample_id)):
        raise ValueError("Training and validation share sample_id values")
    features = [c for c in train if c not in required]
    if not features:
        raise ValueError("No predictors after removing id and label")
    for name, frame in (("train", train), ("validation", valid)):
        if frame[features].isna().any().any():
            raise ValueError(f"Missing values in {name}; imputation was not in the original protocol")
        if not all(pd.api.types.is_numeric_dtype(frame[c]) for c in features):
            raise TypeError(f"Non-numeric predictor in {name}")
        if not np.isfinite(frame[features].to_numpy(dtype=float)).all():
            raise ValueError(f"Infinite feature value in {name}")
    return train, valid, features


def save_csv(rows: object, results_dir: Path, name: str) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    out.to_csv(results_dir / name, index=False, encoding="utf-8")
    LOG.info("Saved %s (%d rows)", name, len(out))
    return out


def run_tree_depth(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], cfg: Config, out: Path) -> None:
    x, y = train[features], train.malignant
    xv, yv = valid[features], valid.malignant
    rows = []
    for depth in (1, 2, 3, 4, None):
        model = DecisionTreeClassifier(max_depth=depth, random_state=cfg.seed).fit(x, y)
        rows.append({
            "max_depth": str(depth) if depth is not None else "unlimited",
            "train_acc": accuracy_score(y, model.predict(x)),
            "validation_acc": accuracy_score(yv, model.predict(xv)),
            "train_auc": roc_auc_score(y, model.predict_proba(x)[:, 1]),
            "validation_auc": roc_auc_score(yv, model.predict_proba(xv)[:, 1]),
            "validation_ba": balanced_accuracy_score(yv, model.predict(xv)),
        })
    save_csv(rows, out, "A_tree_depth.csv")
    roots = []
    for iteration in range(12):
        rng = np.random.default_rng(cfg.seed + iteration + 1)
        sample = rng.choice(len(y), size=len(y), replace=True)
        model = DecisionTreeClassifier(max_depth=2, random_state=cfg.seed).fit(x.iloc[sample], y.iloc[sample])
        root_feature = model.tree_.feature[0]
        roots.append({
            "bootstrap": iteration + 1,
            "root_feature": features[root_feature] if root_feature >= 0 else "leaf",
            "root_threshold": round(float(model.tree_.threshold[0]), 6),
        })
    save_csv(roots, out, "A_bootstrap_roots.csv")


def run_bootstrap(n: int, cfg: Config, out: Path) -> dict[str, float]:
    rng = np.random.default_rng(cfg.seed)
    fraction = np.array([
        len(np.unique(rng.choice(n, n, replace=True))) / n
        for _ in range(cfg.bootstrap_repeats)
    ])
    save_csv({
        "replicate": np.arange(1, len(fraction) + 1),
        "unique_fraction": fraction,
        "oob_fraction": 1 - fraction,
    }, out, "C_bootstrap_1000.csv")  # legacy name retained for report compatibility
    return {
        "bootstrap_unique_mean": float(fraction.mean()),
        "bootstrap_unique_sd": float(fraction.std(ddof=1)) if len(fraction) > 1 else 0.0,
        "bootstrap_unique_min": float(fraction.min()),
        "bootstrap_unique_max": float(fraction.max()),
        "bootstrap_unique_theory": float(1 - (1 - 1 / n) ** n),
    }


def run_models(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], cfg: Config, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    x, y = train[features], train.malignant
    xv, yv = valid[features], valid.malignant
    models = build_models(cfg)
    rows = []
    for name, model in models.items():
        model.fit(x, y)
        labels = model.predict(xv)
        margins = positive_scores(model, xv)
        tn, fp, fn, tp = confusion_matrix(yv, labels, labels=[0, 1]).ravel()
        rows.append({
            "model": name, "roc_auc": roc_auc_score(yv, margins),
            "balanced_accuracy": balanced_accuracy_score(yv, labels),
            "accuracy": accuracy_score(yv, labels),
            "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })
    validation = save_csv(rows, out, "D_F_validation_models.csv")
    development = pd.concat((train, valid), ignore_index=True)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed)
    cv_rows = []
    for name, model in models.items():
        scores = cross_validate(
            clone(model), development[features], development.malignant,
            cv=splitter, n_jobs=cfg.jobs,
            scoring={"auc": "roc_auc", "ba": "balanced_accuracy", "average_precision": "average_precision"},
            error_score="raise",
        )
        cv_rows.append({
            "model": name, "cv_auc_mean": float(scores["test_auc"].mean()),
            "cv_auc_sd": float(scores["test_auc"].std(ddof=1)),
            "cv_ba_mean": float(scores["test_ba"].mean()),
            "cv_ap_mean": float(scores["test_average_precision"].mean()),
        })
    cv_table = save_csv(cv_rows, out, "F_development_cv.csv")
    return validation, cv_table


def run_tree_counts(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], cfg: Config, out: Path) -> None:
    x, y = train[features], train.malignant
    xv, yv = valid[features], valid.malignant
    rows = []
    for n in (1, 10, 50, 100, 300):
        models = {
            "Bagging": BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=cfg.seed), n_estimators=n,
                random_state=cfg.seed, n_jobs=cfg.jobs,
            ),
            "Random forest": RandomForestClassifier(
                n_estimators=n, max_features="sqrt", random_state=cfg.seed, n_jobs=cfg.jobs,
            ),
        }
        for name, model in models.items():
            model.fit(x, y)
            rows.append({
                "model": name, "n_trees": n,
                "auc": roc_auc_score(yv, model.predict_proba(xv)[:, 1]),
                "ba": balanced_accuracy_score(yv, model.predict(xv)),
            })
    save_csv(rows, out, "D_tree_count.csv")


def run_stability(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], cfg: Config, out: Path) -> None:
    x, y = train[features], train.malignant
    xv = valid[features]
    rng = np.random.default_rng(cfg.seed)
    # Crucially all models get EXACTLY the same sampled training row indices.
    sampled_indices = [rng.choice(len(y), len(y), replace=True) for _ in range(cfg.stability_repeats)]
    models = build_models(cfg)
    rows = []
    for name in ("Single tree", "Bagging", "Random forest", "Logistic regression", "RBF-SVM"):
        values = []
        for indices in sampled_indices:
            model = clone(models[name]).fit(x.iloc[indices], y.iloc[indices])
            # Keep logistic regression on the probability scale; only SVM uses a margin.
            # This exactly matches the original stability experiment definition.
            values.append(model.decision_function(xv) if name == "RBF-SVM"
                          else model.predict_proba(xv)[:, 1])
        matrix = np.stack(values, axis=0)
        is_margin = name == "RBF-SVM"
        threshold = 0.0 if is_margin else 0.5
        predictions = matrix >= threshold
        # A flip means a fixed validation record received both labels across repeats.
        positive_fraction = predictions.mean(axis=0)
        rows.append({
            "model": name,
            "prediction_sd_mean": float(matrix.std(axis=0).mean()),
            "prediction_sd_median": float(np.median(matrix.std(axis=0))),
            "flip_rate": float(((positive_fraction > 0) & (positive_fraction < 1)).mean()),
            "score_type": "decision_function (uncalibrated)" if is_margin else "positive_class_probability",
        })
    save_csv(rows, out, "D_G_prediction_stability.csv")


def run_importance(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], cfg: Config, out: Path) -> int:
    x, y = train[features], train.malignant
    xv, yv = valid[features], valid.malignant
    correlation = x.corr().abs()
    pairs = [
        {"feature_1": f, "feature_2": features[j], "abs_r": float(correlation.iloc[i, j])}
        for i, f in enumerate(features)
        for j in range(i + 1, len(features)) if correlation.iloc[i, j] > 0.9
    ]
    table = pd.DataFrame(pairs, columns=["feature_1", "feature_2", "abs_r"])
    table.sort_values("abs_r", ascending=False).to_csv(out / "D_high_correlation_pairs.csv", index=False)
    forest = RandomForestClassifier(n_estimators=300, random_state=cfg.seed, n_jobs=cfg.jobs).fit(x, y)
    permutation = permutation_importance(
        forest, xv, yv, scoring="roc_auc", n_repeats=cfg.permutation_repeats,
        random_state=cfg.seed, n_jobs=cfg.jobs,
    )
    importance = pd.DataFrame({
        "feature": features,
        "gini_importance": forest.feature_importances_,
        "perm_auc_drop_mean": permutation.importances_mean,
        "perm_auc_drop_sd": permutation.importances_std,
    })
    importance.sort_values("gini_importance", ascending=False).to_csv(out / "D_feature_importance.csv", index=False)
    LOG.info("Saved feature importance and %d high-correlation pairs", len(pairs))
    return len(pairs)


def run_boosting(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], cfg: Config, out: Path) -> None:
    x, y = train[features], train.malignant
    xv, yv = valid[features], valid.malignant
    steps = (1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500)
    rows = []
    for rate in (0.03, 0.1, 0.3):
        model = GradientBoostingClassifier(
            n_estimators=cfg.gb_max_estimators, learning_rate=rate, random_state=cfg.seed,
        ).fit(x, y)
        # staged_predict_proba traverses each fitted stage only once.
        for stage, (p_train, p_val) in enumerate(
            zip(model.staged_predict_proba(x), model.staged_predict_proba(xv)), start=1
        ):
            if stage in steps:
                rows.append({
                    "learning_rate": rate, "n_estimators": stage,
                    "train_log_loss": log_loss(y, p_train, labels=[0, 1]),
                    "val_log_loss": log_loss(yv, p_val, labels=[0, 1]),
                    "train_auc": roc_auc_score(y, p_train[:, 1]),
                    "val_auc": roc_auc_score(yv, p_val[:, 1]),
                })
    save_csv(rows, out, "E_gradient_boosting_stages.csv")
    # Teaching example: binary SAMME-style weighted stumps, not sklearn AdaBoost internals.
    weight = np.full(len(y), 1 / len(y))
    trace = []
    for stage in range(1, 6):
        stump = DecisionTreeClassifier(max_depth=1, random_state=cfg.seed + stage)
        stump.fit(x, y, sample_weight=weight)
        wrong = stump.predict(x) != y.to_numpy()
        err = float(np.average(wrong, weights=weight))
        if not 0 < err < 0.5:
            raise RuntimeError(f"Cannot apply binary SAMME weight update at stage {stage}: error={err}")
        alpha = float(np.log((1 - err) / err))
        updated = weight * np.exp(alpha * wrong)
        updated /= updated.sum()
        for i in range(min(10, len(y))):
            trace.append({
                "stage": stage, "sample_id": str(train.sample_id.iloc[i]),
                "true_label": int(y.iloc[i]), "wrong_this_stage": int(wrong[i]),
                "weight_before": weight[i], "weight_after": updated[i],
                "weighted_error": err, "stage_alpha": alpha,
            })
        weight = updated
    save_csv(trace, out, "E_adaboost_weight_trace.csv")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(data_dir: Path, results_dir: Path, cfg: Config) -> None:
    if min(cfg.n_estimators, cfg.bootstrap_repeats, cfg.stability_repeats,
           cfg.permutation_repeats, cfg.gb_max_estimators) <= 0:
        raise ValueError("All repeat counts and estimator counts must be positive")
    results_dir.mkdir(exist_ok=True, parents=True)
    train, valid, features = read_splits(data_dir)
    LOG.info("Loaded %d training rows, %d validation rows, %d features", len(train), len(valid), len(features))
    run_tree_depth(train, valid, features, cfg, results_dir)
    metadata = run_bootstrap(len(train), cfg, results_dir)
    evaluation, cv = run_models(train, valid, features, cfg, results_dir)
    run_tree_counts(train, valid, features, cfg, results_dir)
    run_stability(train, valid, features, cfg, results_dir)
    pairs = run_importance(train, valid, features, cfg, results_dir)
    run_boosting(train, valid, features, cfg, results_dir)
    metadata.update({
        "seed": cfg.seed, "config": asdict(cfg),
        "n_features": len(features), "n_train": len(train), "n_valid": len(valid),
        "n_dev": len(train) + len(valid), "train_positive": int(train.malignant.sum()),
        "valid_positive": int(valid.malignant.sum()), "high_corr_pairs_gt_09": pairs,
        "train_sha256": sha256_file(data_dir / "train.csv"),
        "validation_sha256": sha256_file(data_dir / "validation.csv"),
        "locked_test_labels_used": False,
        "cross_validation_is_development_only": True,
        "validation_best_auc": evaluation.loc[evaluation.roc_auc.idxmax(), "model"],
        "cv_best_auc": cv.loc[cv.cv_auc_mean.idxmax(), "model"],
        "models_evaluated": evaluation.model.tolist(),
        "packages": {"numpy": np.__version__, "pandas": pd.__version__},
    })
    import sklearn
    metadata["packages"]["scikit-learn"] = sklearn.__version__
    (results_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("Completed. Check %s", results_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "student")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--seed", type=int, default=1010)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--stability-repeats", type=int, default=30)
    parser.add_argument("--permutation-repeats", type=int, default=10)
    parser.add_argument("--gb-max-estimators", type=int, default=500)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    config = Config(
        seed=args.seed, bootstrap_repeats=args.bootstrap_repeats,
        stability_repeats=args.stability_repeats, permutation_repeats=args.permutation_repeats,
        gb_max_estimators=args.gb_max_estimators, jobs=args.jobs,
    )
    run(args.data_dir.resolve(), args.results_dir.resolve(), config)


if __name__ == "__main__":
    main()
