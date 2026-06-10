import os
import numpy as np
import json
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib

warnings.filterwarnings('ignore', category=RuntimeWarning)


# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    output_dir   = '/Users/ianchen/Desktop/eegdata/EEGproject/randomforest2_output'
    csv_filename = 'features.csv'

    n_train_subjects = 150
    random_seed      = 42

    # RF hyper-parameters
    rf_n_estimators = 300
    rf_max_depth    = 20      # limit depth — reduces overfit on flat-importance data
    rf_max_features = 'sqrt'
    rf_n_jobs       = -1
    rf_class_weight = 'balanced'

    # Subject normalisation
    # Features with std == 0 across a subject's segments get filled with 0
    norm_fill_value = 0.0

    def __init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"✓ Output directory: {self.output_dir}")

    @property
    def csv_path(self):
        return os.path.join(self.output_dir, self.csv_filename)


# ============================================================
# DATA LOADING
# ============================================================
def load_csv(csv_path: str):
    print(f"  Loading CSV : {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows × {df.shape[1]} columns")

    meta_cols = {'subject_id', 'label', 'filepath'}
    feat_cols = [c for c in df.columns if c not in meta_cols]

    X           = np.nan_to_num(df[feat_cols].values.astype(float), nan=0.0)
    y           = df['label'].values.astype(int)
    subject_ids = df['subject_id'].values.astype(str)

    return X, y, feat_cols, subject_ids


# ============================================================
# SUBJECT-LEVEL Z-SCORE NORMALISATION  ← the key improvement
# ============================================================
def subject_normalize(X: np.ndarray,
                      subject_ids: np.ndarray,
                      fill_value: float = 0.0) -> np.ndarray:
    X_norm   = np.empty_like(X, dtype=np.float64)
    subjects = np.unique(subject_ids)

    for sid in subjects:
        idx  = np.where(subject_ids == sid)[0]
        mu   = X[idx].mean(axis=0)
        std  = X[idx].std(axis=0, ddof=1)
        std  = np.where(std == 0, 1.0, std)   # avoid /0; result will be 0
        X_norm[idx] = (X[idx] - mu) / std

    # Replace NaN/Inf that may arise from edge cases
    X_norm = np.nan_to_num(X_norm, nan=fill_value,
                           posinf=fill_value, neginf=fill_value)

    return X_norm


# ============================================================
# SUBJECT-LEVEL SPLIT
# ============================================================
def subject_level_split(subject_ids, n_train: int):
    unique_subs = sorted(set(subject_ids))
    train_subs  = set(unique_subs[:n_train])
    test_subs   = set(unique_subs[n_train:])

    train_idx = np.where([s in train_subs for s in subject_ids])[0]
    test_idx  = np.where([s in test_subs  for s in subject_ids])[0]

    print(f"  Train subjects : {len(train_subs):>4}  ({len(train_idx)} samples)")
    print(f"  Test  subjects : {len(test_subs):>4}  ({len(test_idx)} samples)")
    return train_idx, test_idx


# ============================================================
# PLOTS
# ============================================================
def plot_confusion(y_true, y_pred, title, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Stage 1', 'Stage 2'],
                yticklabels=['Stage 1', 'Stage 2'],
                cbar_kws={'label': 'Count'})

    # Per-class accuracy in cell annotations
    for i in range(2):
        row_sum = cm[i].sum()
        pct = cm[i, i] / row_sum * 100 if row_sum > 0 else 0
        ax.text(i + 0.5, i + 0.72, f'({pct:.1f}%)',
                ha='center', va='center', fontsize=10,
                color='white' if cm[i, i] > cm.max() / 2 else 'black')

    ax.set_title(title)
    ax.set_ylabel('True Stage')
    ax.set_xlabel('Predicted Stage')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out_path}")


def plot_roc(y_true, y_score, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, lw=2, label=f'ROC  AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve — Stage 1 vs Stage 2 (Subject-Normalised)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out_path}")


def plot_feature_importance(importances, feat_names, top_n, out_path):
    n     = min(top_n, len(feat_names))
    order = np.argsort(importances)[::-1][:n]
    plt.figure(figsize=(10, max(4, n * 0.28)))
    plt.barh(range(n), importances[order][::-1], color='steelblue')
    plt.yticks(range(n), [feat_names[i] for i in order[::-1]], fontsize=8)
    plt.xlabel('Mean Gini Importance (normalised)')
    plt.title(f'Top {n} Features — RF Subject-Normalised (Gini Importance)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out_path}")


def plot_importance_comparison(imp_raw, imp_norm, feat_names, top_n, out_path):
    """Side-by-side bar chart: raw vs normalised feature importances."""
    n     = min(top_n, len(feat_names))
    order = np.argsort(imp_norm)[::-1][:n]

    x     = np.arange(n)
    names = [feat_names[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(18, max(4, n * 0.28)), sharey=True)
    for ax, imp, title in zip(axes,
                               [imp_raw[order], imp_norm[order]],
                               ['Raw features', 'Subject-normalised features']):
        ax.barh(x, imp[::-1], color='steelblue')
        ax.set_yticks(x)
        ax.set_yticklabels(names[::-1], fontsize=7)
        ax.set_xlabel('Gini Importance')
        ax.set_title(title)

    fig.suptitle(f'Top {n} Features: Raw vs Subject-Normalised', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("Random Forest — Subject-Normalised EEG Features")
    print("  Normalisation : per-subject z-score (mean=0, std=1)")
    print("  Model         : RandomForestClassifier (max_depth=20)")
    print("=" * 60)

    cfg = Config()

    # ── 1. Load ──────────────────────────────────────────────
    if not os.path.exists(cfg.csv_path):
        raise FileNotFoundError(
            f"CSV not found: {cfg.csv_path}\n"
            "Run extract_features_to_csv.py first."
        )

    print("\n" + "=" * 60)
    print("LOADING FEATURES")
    print("=" * 60)
    X, y, feat_names, subject_ids = load_csv(cfg.csv_path)
    print(f"  Feature matrix : {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Unique subjects: {len(np.unique(subject_ids))}")
    print(f"  Class balance  : Stage1={np.sum(y==0)}  Stage2={np.sum(y==1)}")

    # ── 2. Subject-level split (BEFORE normalisation) ────────
    # Important: split first, normalise second.
    # Each subject is normalised using only its OWN segments —
    # no information leaks from test subjects into training.
    print("\n" + "=" * 60)
    print("SUBJECT-LEVEL SPLIT")
    print("=" * 60)
    train_idx, test_idx = subject_level_split(subject_ids, cfg.n_train_subjects)

    # ── 3. Subject-level normalisation ───────────────────────
    print("\n" + "=" * 60)
    print("SUBJECT NORMALISATION  (z-score within each subject)")
    print("=" * 60)
    print("  Normalising … ", end='', flush=True)
    X_norm = subject_normalize(X, subject_ids, fill_value=cfg.norm_fill_value)
    print("done")

    # Quick sanity check: mean and std per feature across all samples
    mu_check  = X_norm.mean(axis=0)
    std_check = X_norm.std(axis=0)
    print(f"  Post-norm  feature mean : {mu_check.mean():.4f}  "
          f"(ideal ≈ 0.00)")
    print(f"  Post-norm  feature std  : {std_check.mean():.4f}  "
          f"(ideal ≈ 1.00)")

    X_train, y_train = X_norm[train_idx], y[train_idx]
    X_test,  y_test  = X_norm[test_idx],  y[test_idx]
    print(f"\n  Train : {X_train.shape}  labels={np.bincount(y_train)}")
    print(f"  Test  : {X_test.shape}   labels={np.bincount(y_test)}")

    # ── 4. RF model ───────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators  = cfg.rf_n_estimators,
        max_depth     = cfg.rf_max_depth,
        max_features  = cfg.rf_max_features,
        class_weight  = cfg.rf_class_weight,
        random_state  = cfg.random_seed,
        n_jobs        = cfg.rf_n_jobs,
    )

    # ── 5. 5-fold CV ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("5-FOLD CROSS-VALIDATION  (training subjects only)")
    print("=" * 60)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.random_seed)
    cv_scores = cross_val_score(
        rf, X_train, y_train,
        cv=cv, scoring='balanced_accuracy',
        n_jobs=cfg.rf_n_jobs,
    )
    print(f"  Balanced accuracy : {cv_scores.mean()*100:.2f}% "
          f"± {cv_scores.std()*100:.2f}%  (per fold: "
          + ", ".join(f"{s*100:.1f}%" for s in cv_scores) + ")")

    # ── 6. Train ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING ON FULL TRAIN SET")
    print("=" * 60)
    rf.fit(X_train, y_train)
    print("  ✓ RF trained")

    # ── 7. Test evaluation ───────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION  (held-out subjects)")
    print("=" * 60)
    y_pred  = rf.predict(X_test)
    y_score = rf.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred,
                                target_names=['Stage 1', 'Stage 2']))
    auc = roc_auc_score(y_test, y_score)
    print(f"  ROC-AUC : {auc:.4f}")

    # Compare vs raw (unnormalised) RF for reference
    print("\n  ── Comparison ──────────────────────────────────────")
    print("  Previous RF (no normalisation) : AUC ≈ 0.678  Acc ≈ 0.65")
    print(f"  This RF   (subject-normalised) : AUC = {auc:.3f}  "
          f"Acc = {(y_pred == y_test).mean():.3f}")

    # ── 8. Feature importance ────────────────────────────────
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE  (Gini, subject-normalised model)")
    print("=" * 60)
    imp_norm = rf.feature_importances_
    top_idx  = np.argsort(imp_norm)[::-1][:10]
    print("  Top 10 features:")
    for rank, i in enumerate(top_idx, 1):
        print(f"    {rank:>2}. {feat_names[i]:<50}  {imp_norm[i]:.5f}")

    # Also train a quick raw RF for the comparison plot
    print("\n  Training raw (unnormalised) RF for comparison plot …")
    rf_raw = RandomForestClassifier(
        n_estimators = cfg.rf_n_estimators,
        max_depth    = cfg.rf_max_depth,
        max_features = cfg.rf_max_features,
        class_weight = cfg.rf_class_weight,
        random_state = cfg.random_seed,
        n_jobs       = cfg.rf_n_jobs,
    )
    rf_raw.fit(X[train_idx], y_train)
    imp_raw = rf_raw.feature_importances_
    print("  ✓ done")

    # ── 9. Save outputs ───────────────────────────────────────
    plot_confusion(
        y_test, y_pred,
        title    = "RF Subject-Normalised — Stage 1 vs Stage 2 (Test Set)",
        out_path = os.path.join(cfg.output_dir, 'rf_norm_confusion_matrix.png'),
    )
    plot_roc(
        y_test, y_score,
        out_path = os.path.join(cfg.output_dir, 'rf_norm_roc_curve.png'),
    )
    plot_feature_importance(
        imp_norm, feat_names, top_n=30,
        out_path = os.path.join(cfg.output_dir, 'rf_norm_feature_importance.png'),
    )
    plot_importance_comparison(
        imp_raw, imp_norm, feat_names, top_n=20,
        out_path = os.path.join(cfg.output_dir, 'rf_norm_importance_comparison.png'),
    )

    model_path = os.path.join(cfg.output_dir, 'rf_norm_model.joblib')
    joblib.dump(rf, model_path)
    print(f"  ✓ Model saved : {model_path}")

    summary = {
        'task'                    : 'Stage 1 vs Stage 2 — RF, subject-normalised',
        'normalisation'           : 'per-subject z-score (applied before split exposure)',
        'n_features'              : int(X.shape[1]),
        'n_train_subjects'        : int(cfg.n_train_subjects),
        'n_train_samples'         : int(len(X_train)),
        'n_test_samples'          : int(len(X_test)),
        'cv_balanced_acc_mean'    : float(cv_scores.mean()),
        'cv_balanced_acc_std'     : float(cv_scores.std()),
        'test_roc_auc'            : float(auc),
        'test_accuracy'           : float((y_pred == y_test).mean()),
        'rf_n_estimators'         : cfg.rf_n_estimators,
        'rf_max_depth'            : cfg.rf_max_depth,
        'rf_max_features'         : cfg.rf_max_features,
        'previous_rf_auc_no_norm' : 0.678,
    }
    summary_path = os.path.join(cfg.output_dir, 'rf_norm_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Summary saved : {summary_path}")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"  Output folder : {cfg.output_dir}")
    print(f"  rf_norm_model.joblib")
    print(f"  rf_norm_confusion_matrix.png")
    print(f"  rf_norm_roc_curve.png")
    print(f"  rf_norm_feature_importance.png")
    print(f"  rf_norm_importance_comparison.png   ← raw vs normalised side-by-side")
    print(f"  rf_norm_summary.json")
    print("=" * 60)


if __name__ == "__main__":
    main()