"""
Лабораторная работа №2 (вариант): LSTM-автоэнкодер для поиска неисправностей.
Вход — окно из нескольких подряд идущих шагов телеметрии. Автоэнкодер учится
восстанавливать окна нормальной работы; окно с высокой ошибкой восстановления
считается неисправным (предсказание относится к последнему шагу окна).
Порог подбирается на обучающей части: отложенные нормальные окна + известные неисправности.
"""
import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from DataWork import SatelliteDataset
from model import build_model

CLASS_NAMES = ["норма", "неисправность"]

LOGGER = logging.getLogger("training")

def metrics(y_true, proba, thr=0.5):
    y_pred = (proba >= thr).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, proba)) if y_true.sum() else float("nan"),
    }


def make_windows(df, feature_names, window):
    """Для каждого шага — окно из `window` последних подряд идущих шагов (для LSTM).
    Окно не пересекает разрывы по времени (шаги без телеметрии); в начале участка
    недостающие позиции дополняются первым доступным шагом."""
    Xs_all = df[feature_names].values
    t = pd.to_numeric(df["time"]).values
    order = np.argsort(t, kind="stable")
    ts, Xs = t[order], Xs_all[order]
    run_start = np.zeros(len(ts), dtype=int)
    for k in range(1, len(ts)):
        run_start[k] = run_start[k - 1] if ts[k] - ts[k - 1] == 1 else k
    W = np.zeros((len(ts), window, Xs.shape[1]))
    for k in range(len(ts)):
        idx = list(range(max(run_start[k], k - window + 1), k + 1))
        idx = [idx[0]] * (window - len(idx)) + idx
        W[k] = Xs[idx]
    out = np.zeros_like(W)
    out[order] = W
    return out


def best_threshold(y_true, proba):
    """Порог, максимизирующий F1 по кривой precision-recall (при равенстве берём больший)."""
    p, r, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    return float(thr[np.flatnonzero(f1 == f1.max())[-1]])

def ae_run(W_tr, y_tr, W_te, epochs, batch_size, seed, patience=20):
    """W_* — окна формы (n, window, features). Возвращает оценки аномальности теста, порог, модель."""
    n_feat = W_tr.shape[2]
    normal = np.where(y_tr == 0)[0]
    faults = np.where(y_tr == 1)[0]
    fit_idx, cal_idx = train_test_split(normal, test_size=0.2, random_state=seed)

    scaler = StandardScaler().fit(W_tr[fit_idx].reshape(-1, n_feat))
    prep = lambda W: scaler.transform(W.reshape(-1, n_feat)).reshape(W.shape)
    W_fit, W_cal = prep(W_tr[fit_idx]), prep(W_tr[cal_idx])

    tf.keras.utils.set_random_seed(seed)
    model = build_model(W_tr.shape[1], n_feat)
    hist = model.fit(
        W_fit, W_fit, validation_data=(W_cal, W_cal),
        epochs=epochs, batch_size=batch_size, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True)],
    )

    def score(W):
        Z = prep(W)
        return ((Z - model.predict(Z, verbose=0)) ** 2).mean(axis=(1, 2))

    cal_all = np.concatenate([cal_idx, faults])
    thr = best_threshold(y_tr[cal_all], score(W_tr[cal_all]))
    return score(W_te), thr, model, scaler, hist


def run(output_dir, data_dir=None, use_prev=False, window=5, epochs=200, batch_size=32,
        test_size=0.3, folds=3, random_seed=42):
    output_dir.mkdir(parents=True, exist_ok=True)
    ds = SatelliteDataset(data_dir, use_prev)
    df, feature_names = ds.get_data()
    y = df["label"].values
    W = make_windows(df, feature_names, window)
    LOGGER.info("Шагов: %d (норма %d, неисправность %d), признаков: %d, окно: %d; исключено без телеметрии: %d",
                len(y), int((y == 0).sum()), int(y.sum()), len(feature_names), window, ds.dropped_no_telemetry)

    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=test_size, stratify=y, random_state=random_seed)
    s_te, thr, model, scaler, hist = ae_run(W[tr_idx], y[tr_idx], W[te_idx], epochs, batch_size, random_seed)
    y_te = y[te_idx]
    y_pred = (s_te >= thr).astype(int)
    test_m = metrics(y_te, s_te, thr)
    cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
    report = classification_report(y_te, y_pred, labels=[0, 1], target_names=CLASS_NAMES, zero_division=0)
    LOGGER.info("Порог ошибки восстановления: %.4f", thr)
    LOGGER.info("Тест: %s", test_m)
    print(report)

    k = max(2, min(folds, int(y.sum())))
    rows = []
    for f, (a, b) in enumerate(StratifiedKFold(k, shuffle=True, random_state=random_seed).split(W, y), 1):
        s, t_, *_ = ae_run(W[a], y[a], W[b], epochs, batch_size, random_seed + f)
        rows.append(metrics(y[b], s, t_))
    cv = pd.DataFrame(rows, index=[f"fold_{i}" for i in range(1, k + 1)])
    cv.to_csv(output_dir / "cv_metrics.csv")
    LOGGER.info("Кросс-валидация (%d фолда), среднее:\n%s", k, cv.mean().round(4).to_string())

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(hist.history["loss"], label="train (норма)"); ax.plot(hist.history["val_loss"], label="val (норма)")
    ax.set_title("MSE LSTM-автоэнкодера"); ax.set_xlabel("Эпоха"); ax.legend()
    fig.tight_layout(); fig.savefig(output_dir / "learning_curves.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues", colorbar=False)
    fig.tight_layout(); fig.savefig(output_dir / "confusion_matrix.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ls = np.log10(s_te + 1e-9)
    ax.hist(ls[y_te == 0], bins=30, alpha=0.7, label="норма")
    ax.hist(ls[y_te == 1], bins=30, alpha=0.7, label="неисправность")
    ax.axvline(np.log10(thr + 1e-9), color="k", linestyle="--", label="порог")
    ax.set_xlabel("log10(ошибка восстановления)"); ax.set_title("Распределение ошибок на тесте"); ax.legend()
    fig.tight_layout(); fig.savefig(output_dir / "score_distribution.png", dpi=150); plt.close(fig)

    te = df.iloc[te_idx][["time", "label"]].copy()
    te["predicted"] = y_pred
    te["score"] = s_te
    te[te["label"] != te["predicted"]].to_csv(output_dir / "misclassified_test_steps.csv", index=False)
    te.to_csv(output_dir / "test_predictions.csv", index=False)

    model.save(output_dir / "lstm_autoencoder.keras")
    joblib.dump({"scaler": scaler, "feature_names": feature_names, "threshold": thr, "window": window},
                output_dir / "preprocessing.joblib")
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    results = {
        "model": "lstm_autoencoder", "window": window, "time_steps": int(len(y)), "faults": int(y.sum()),
        "features": len(feature_names), "dropped_no_telemetry": ds.dropped_no_telemetry,
        "train_steps": int(len(tr_idx)), "test_steps": int(len(te_idx)), "test_faults": int(y_te.sum()),
        "baseline_accuracy_always_normal": float((y == 0).mean()), "threshold": thr,
        "test": test_m, "cv_mean": cv.mean().to_dict(), "cv_std": cv.std().fillna(0).to_dict(),
        "epochs_max": epochs, "random_seed": random_seed,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    LOGGER.info("Артефакты сохранены в %s", output_dir.resolve())
    return results


def parse_args():
    ap = argparse.ArgumentParser(description="LSTM-автоэнкодер: поиск неисправностей как аномалий")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--use-prev", action="store_true")
    ap.add_argument("--window", type=int, default=5, help="Длина окна (шагов)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts_lstm_ae")
    return ap.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    a = parse_args()
    run(a.output_dir, a.data_dir, a.use_prev, a.window, a.epochs, a.batch_size, folds=a.folds, random_seed=a.seed)
