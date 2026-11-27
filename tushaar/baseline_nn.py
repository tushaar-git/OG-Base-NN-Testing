"""MLP baseline on ECG5000 (data/dataset_).

Run from anywhere:  .venv\\Scripts\\python.exe tushaar\\baseline_nn.py
Outputs go to tushaar/results/.
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import tensorflow as tf
import keras
from scipy.io import arff
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "data" / "dataset_"
OUT = HERE / "results"

SPLIT_SEED = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15
MAIN_SEED = 42
REPEAT_SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
THRESHOLD = 0.5
N_FEATURES = 140
EXPECTED_PARAMS = 11137

summary_lines = []


def log(text=""):
    print(text)
    summary_lines.append(str(text))


# ---------------------------------------------------------------- determinism
try:
    tf.config.experimental.enable_op_determinism()
    DETERMINISM_ENABLED, DETERMINISM_MSG = True, "enabled"
except Exception as e:  # report, don't hide
    DETERMINISM_ENABLED, DETERMINISM_MSG = False, f"FAILED: {type(e).__name__}: {e}"


# ----------------------------------------------------------------------- data
def load_data():
    raw, meta = arff.loadarff(DATA_PATH)
    df = pd.DataFrame(raw)
    feature_cols = [f"col_{i}" for i in range(N_FEATURES)]
    if list(df.columns) != feature_cols + ["class"]:
        raise ValueError(f"Unexpected ARFF columns: {list(df.columns)[:3]}...{list(df.columns)[-3:]}")
    y_float = df["class"].to_numpy(dtype=float)
    y = y_float.astype(int)
    if not np.array_equal(y, y_float) or set(np.unique(y)) != {0, 1}:
        raise ValueError(f"class is not clean 0/1: {np.unique(y_float)}")
    X = df[feature_cols].to_numpy(dtype=np.float32)
    return X, y


def make_split(y):
    idx = np.arange(len(y))
    train_idx, temp_idx = train_test_split(
        idx, test_size=VAL_FRAC + TEST_FRAC, stratify=y, random_state=SPLIT_SEED)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
        stratify=y[temp_idx], random_state=SPLIT_SEED)
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------- model
def build_model(seed):
    keras.utils.set_random_seed(seed)
    model = keras.Sequential([
        keras.Input(shape=(N_FEATURES,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(loss="binary_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  metrics=["accuracy"])
    return model


def dense_weights(model):
    out = {}
    for i, layer in enumerate(model.layers):
        kernel, bias = layer.get_weights()
        out[f"dense{i}_kernel"] = kernel
        out[f"dense{i}_bias"] = bias
    return out


def run(seed, data, save_extras=False):
    Xtr, ytr, Xva, yva, Xte, yte, test_idx = data
    model = build_model(seed)
    if save_extras:
        np.savez(OUT / f"initial_weights_seed{seed}.npz", **dense_weights(model))

    t0 = time.perf_counter()
    hist = model.fit(Xtr, ytr, epochs=EPOCHS, batch_size=BATCH_SIZE,
                     validation_data=(Xva, yva), shuffle=True, verbose=0)
    train_time = time.perf_counter() - t0

    h = pd.DataFrame(hist.history)
    h.insert(0, "epoch", np.arange(1, len(h) + 1))
    h = h[["epoch", "loss", "accuracy", "val_loss", "val_accuracy"]]
    h.to_csv(OUT / f"history_seed{seed}.csv", index=False)

    test_loss, test_acc = model.evaluate(Xte, yte, batch_size=BATCH_SIZE, verbose=0)
    prob = model.predict(Xte, batch_size=BATCH_SIZE, verbose=0).ravel()
    pred = (prob > THRESHOLD).astype(int)
    pd.DataFrame({"row_index": test_idx, "true_label": yte,
                  "pred_prob": prob, "pred_label": pred}).to_csv(
        OUT / f"test_predictions_seed{seed}.csv", index=False)

    tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
    best = int(h["val_loss"].idxmin())
    row = dict(
        seed=seed, test_loss=test_loss, test_accuracy=test_acc,
        precision=precision_score(yte, pred, zero_division=0),
        recall=recall_score(yte, pred, zero_division=0),
        f1=f1_score(yte, pred, zero_division=0),
        roc_auc=roc_auc_score(yte, prob),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        final_train_loss=h["loss"].iloc[-1], final_train_accuracy=h["accuracy"].iloc[-1],
        final_val_loss=h["val_loss"].iloc[-1], final_val_accuracy=h["val_accuracy"].iloc[-1],
        best_val_loss=h["val_loss"].iloc[best], best_val_epoch=int(h["epoch"].iloc[best]),
        train_time_s=train_time,
    )

    if save_extras:
        np.savez(OUT / f"final_weights_seed{seed}.npz", **dense_weights(model))
        model.save(OUT / f"model_seed{seed}.keras")
        plot_curves(h, OUT / f"curves_seed{seed}.png", seed)
    return model, row


def plot_curves(h, path, seed):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, label in [(axes[0], "loss", "Loss (binary cross-entropy)"),
                           (axes[1], "accuracy", "Accuracy")]:
        ax.plot(h["epoch"], h[key], label="train")
        ax.plot(h["epoch"], h[f"val_{key}"], label="val")
        ax.set_xlabel("Epoch"); ax.set_ylabel(label); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle(f"MLP baseline, seed {seed}")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def cpu_name():
    if sys.platform == "win32":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


# ----------------------------------------------------------------------- main
def main():
    OUT.mkdir(exist_ok=True)
    log(f"Op determinism: {DETERMINISM_MSG}")

    X, y = load_data()
    train_idx, val_idx, test_idx = make_split(y)
    np.savez(OUT / "split_indices.npz", train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    mean = X[train_idx].mean(axis=0)
    std = X[train_idx].std(axis=0)  # population std (ddof=0)
    if np.any(std == 0):
        raise ValueError("Zero-variance feature in train set")
    np.savez(OUT / "scaler_stats.npz", mean=mean, std=std)
    Xs = (X - mean) / std

    log(f"Data: {X.shape[0]} rows x {X.shape[1]} features")
    for name, ix in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        log(f"  {name:5s}: {len(ix):4d} rows, class1 = {y[ix].mean():.4f}")
    data = (Xs[train_idx], y[train_idx], Xs[val_idx], y[val_idx], Xs[test_idx], y[test_idx], test_idx)

    rows = []
    log(f"\n=== Run 1: seed {MAIN_SEED} ===")
    model, row = run(MAIN_SEED, data, save_extras=True)
    rows.append(row)

    s = []
    model.summary(print_fn=lambda line, **kw: s.append(line))
    log("\n".join(s))
    n_params = model.count_params()
    flag = "OK" if n_params == EXPECTED_PARAMS else f"MISMATCH (expected {EXPECTED_PARAMS:,})"
    log(f"Total parameters: {n_params:,}  [{flag}]")
    log(f"Seed {MAIN_SEED}: final train loss {row['final_train_loss']:.4f}, acc {row['final_train_accuracy']:.4f} | "
        f"final val loss {row['final_val_loss']:.4f}, acc {row['final_val_accuracy']:.4f}")
    log(f"Seed {MAIN_SEED}: lowest val_loss {row['best_val_loss']:.4f} at epoch {row['best_val_epoch']}")
    log(f"Seed {MAIN_SEED}: training time {row['train_time_s']:.1f} s")

    log("\n=== Run 2: seeds " + ", ".join(map(str, REPEAT_SEEDS)) + " ===")
    for seed in REPEAT_SEEDS:
        print(f"training seed {seed} ...", flush=True)
        _, r = run(seed, data)
        rows.append(r)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "metrics.csv", index=False)

    log("\n=== Per-run test metrics ===")
    for r in rows:
        log(f"seed {r['seed']:>2}: loss {r['test_loss']:.4f}  acc {r['test_accuracy']:.4f}  "
            f"prec {r['precision']:.4f}  rec {r['recall']:.4f}  f1 {r['f1']:.4f}  "
            f"auc {r['roc_auc']:.4f}  time {r['train_time_s']:.1f}s")
        log(f"         confusion [[TN, FP],[FN, TP]] = [[{r['tn']}, {r['fp']}],[{r['fn']}, {r['tp']}]]")

    rep = metrics[metrics["seed"].isin(REPEAT_SEEDS)].drop(columns="seed")
    agg = pd.DataFrame({"mean": rep.mean(), "std": rep.std(ddof=1)}).T
    agg.index.name = "stat"
    agg.to_csv(OUT / "metrics_aggregate.csv")

    log(f"\n=== Across seeds {REPEAT_SEEDS} (mean ± sample std, ddof=1) ===")
    for col in ["test_accuracy", "f1", "roc_auc", "train_time_s"]:
        log(f"  {col:14s}: {agg.loc['mean', col]:.4f} ± {agg.loc['std', col]:.4f}")
    log(f"\nOp determinism enabled: {DETERMINISM_ENABLED} ({DETERMINISM_MSG})")

    config = dict(
        data_path=str(DATA_PATH.relative_to(HERE.parent)),
        architecture=["Input(shape=(140,))", "Dense(64, relu)", "Dense(32, relu)", "Dense(1, sigmoid)"],
        total_params=n_params,
        kernel_initializer="glorot_uniform (Keras default)", bias_initializer="zeros (Keras default)",
        regularisation="none (no dropout, no batch norm, no weight penalties)",
        optimizer="Adam", learning_rate=LEARNING_RATE, other_optimizer_args="Keras defaults",
        loss="binary_crossentropy", metrics=["accuracy"],
        batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True, callbacks="none",
        split={"train": TRAIN_FRAC, "val": VAL_FRAC, "test": TEST_FRAC,
               "method": "sklearn train_test_split twice (70/30, then 30 -> 50/50), stratified on class",
               "split_seed": SPLIT_SEED,
               "sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)}},
        scaling="per-feature z-score, mean and population std (ddof=0) from train set only",
        threshold=THRESHOLD, threshold_rule="pred = prob > 0.5",
        main_seed=MAIN_SEED, repeat_seeds=REPEAT_SEEDS,
        seeding="keras.utils.set_random_seed(seed) immediately before building each model",
        aggregate_std="sample std (ddof=1) over repeat seeds",
        op_determinism={"enabled": DETERMINISM_ENABLED, "message": DETERMINISM_MSG},
        versions={"python": platform.python_version(), "numpy": np.__version__,
                  "pandas": pd.__version__, "tensorflow": tf.__version__, "keras": keras.__version__,
                  "scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
                  "matplotlib": matplotlib.__version__},
        os=platform.platform(), cpu=cpu_name(),
        gpu_devices=[d.name for d in tf.config.list_physical_devices("GPU")],
    )
    (OUT / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (OUT / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
