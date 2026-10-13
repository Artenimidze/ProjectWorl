"""
Обучение регрессионной модели оценки ёмкости батареи.
"""
import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from DataWork import Datasets
from model import build_model


LOGGER = logging.getLogger("training")
FEATURE_NAMES = ["voltage", "current", "temperature"]
TARGET_NAME = "capacity"


def train_model(
    output_dir: Path,
    epochs: int = 200,
    batch_size: int = 32,
    patience: int = 20,
    random_seed: int = 42,
) -> dict:
    """Обучает модель на B0005-B0007 и оценивает её на B0018."""
    if epochs < 1:
        raise ValueError("Число эпох должно быть больше нуля")
    if batch_size < 1:
        raise ValueError("Размер batch должен быть больше нуля")
    if patience < 1:
        raise ValueError("Patience должно быть больше нуля")

    tf.keras.utils.set_random_seed(random_seed)
    df_all = Datasets.NASA_BATTERY_DATASET.value.get_data()
    validation_battery = "B0007"
    training_batteries = ["B0005", "B0006", "B0007"]
    validation_train = df_all[df_all["battery"].isin(["B0005", "B0006"])]
    validation = df_all[df_all["battery"] == validation_battery]
    final_train = df_all[df_all["battery"].isin(training_batteries)]
    test = df_all[df_all["battery"] == "B0018"]
    if validation_train.empty or validation.empty or final_train.empty or test.empty:
        raise ValueError(
            "Для обучения нужны B0005, B0006, B0007 и тестовая батарея B0018"
        )

    X_validation_train = validation_train[FEATURE_NAMES].values
    y_validation_train = validation_train[TARGET_NAME].values
    X_validation = validation[FEATURE_NAMES].values
    y_validation = validation[TARGET_NAME].values

    validation_feature_scaler = MinMaxScaler()
    validation_target_scaler = MinMaxScaler()
    X_validation_train_scaled = validation_feature_scaler.fit_transform(X_validation_train)
    X_validation_scaled = validation_feature_scaler.transform(X_validation)
    y_validation_train_scaled = validation_target_scaler.fit_transform(
        y_validation_train.reshape(-1, 1)
    ).ravel()
    y_validation_scaled = validation_target_scaler.transform(
        y_validation.reshape(-1, 1)
    ).ravel()

    selection_model = build_model(input_dim=len(FEATURE_NAMES))
    selection_history = selection_model.fit(
        X_validation_train_scaled,
        y_validation_train_scaled,
        validation_data=(X_validation_scaled, y_validation_scaled),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        )],
        verbose=1,
    )
    best_epoch = int(np.argmin(selection_history.history["val_loss"]) + 1)

    X_final_train = final_train[FEATURE_NAMES].values
    y_final_train = final_train[TARGET_NAME].values
    X_test = test[FEATURE_NAMES].values
    y_test = test[TARGET_NAME].values
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    X_final_train_scaled = feature_scaler.fit_transform(X_final_train)
    X_test_scaled = feature_scaler.transform(X_test)
    y_final_train_scaled = target_scaler.fit_transform(
        y_final_train.reshape(-1, 1)
    ).ravel()

    tf.keras.utils.set_random_seed(random_seed)
    model = build_model(input_dim=len(FEATURE_NAMES))
    final_history = model.fit(
        X_final_train_scaled,
        y_final_train_scaled,
        epochs=best_epoch,
        batch_size=batch_size,
        verbose=1,
    )
    predictions_scaled = model.predict(X_test_scaled, verbose=0).reshape(-1, 1)
    predictions = target_scaler.inverse_transform(predictions_scaled).ravel()
    test_mae = float(mean_absolute_error(y_test, predictions))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    test_r2 = float(r2_score(y_test, predictions))
    validation_predictions_scaled = selection_model.predict(X_validation_scaled, verbose=0)
    validation_predictions = validation_target_scaler.inverse_transform(
        validation_predictions_scaled.reshape(-1, 1)
    ).ravel()
    validation_mae = float(mean_absolute_error(y_validation, validation_predictions))
    validation_rmse = float(np.sqrt(mean_squared_error(y_validation, validation_predictions)))
    validation_r2 = float(r2_score(y_validation, validation_predictions))

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "battery_capacity_model.keras")
    joblib.dump(
        {
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
            "feature_names": FEATURE_NAMES,
            "target_name": TARGET_NAME,
        },
        output_dir / "preprocessing.joblib",
    )
    pd.DataFrame(selection_history.history).to_csv(
        output_dir / "validation_history.csv", index=False
    )
    pd.DataFrame(final_history.history).to_csv(
        output_dir / "training_history.csv", index=False
    )

    results = {
        "validation_train_samples": int(len(validation_train)),
        "validation_samples": int(len(validation)),
        "final_train_samples": int(len(final_train)),
        "test_samples": int(len(X_test)),
        "epochs_trained": len(final_history.history["loss"]),
        "best_epoch": best_epoch,
        "validation_battery": validation_battery,
        "validation_mae": validation_mae,
        "validation_rmse": validation_rmse,
        "validation_r2": validation_r2,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "random_seed": random_seed,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(results, metrics_file, indent=2, ensure_ascii=False)

    LOGGER.info("Итоговая оценка на B0018: MAE=%.4f, RMSE=%.4f, R2=%.4f", test_mae, test_rmse, test_r2)
    LOGGER.info("Артефакты обучения сохранены в %s", output_dir.resolve())
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Обучает регрессионную модель для NASA Battery Dataset"
    )
    parser.add_argument("--epochs", type=int, default=200, help="Максимальное число эпох")
    parser.add_argument("--batch-size", type=int, default=32, help="Размер batch")
    parser.add_argument("--patience", type=int, default=20, help="Эпох без улучшения до остановки")
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимости")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
        help="Папка для сохранения модели и результатов",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    train_model(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        random_seed=args.seed,
    )
