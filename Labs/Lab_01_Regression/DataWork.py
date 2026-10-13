"""
DataWork.py — подготовка данных NASA PCoE Battery Dataset.

Интерфейс, которого ждёт train.py:
    from DataWork import Datasets
    df = Datasets.NASA_BATTERY_DATASET.value.get_data()

get_data() возвращает pandas.DataFrame: одна строка = один разрядный цикл.
Обязательные колонки:
    battery      — имя батареи (B0005, B0006, B0007, B0018)
    voltage      — среднее измеренное напряжение за цикл разряда, В
    current      — средний измеренный ток за цикл разряда, А
    temperature  — средняя температура за цикл разряда, °C
    capacity     — ёмкость цикла, Ач (целевая переменная)
Дополнительные колонки (для экспериментов, train.py их игнорирует,
пока их нет в FEATURE_NAMES):
    cycle, ambient_temp, v_min, v_std, t_max, t_min, duration, time_below_3_5V
"""
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

DEFAULT_BATTERIES = ("B0005", "B0006", "B0007", "B0018")


class NasaBatteryDataset:
    """Чтение .mat файлов NASA PCoE и агрегация признаков по циклам разряда."""

    def __init__(self, data_dir=None, batteries=DEFAULT_BATTERIES):
        # .mat файлы лежат рядом с этим модулем (корень репозитория)
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parent
        self.batteries = tuple(batteries)
        self._cache = None

    # ------------------------------------------------------------------
    def _load_battery(self, name: str) -> pd.DataFrame:
        path = self.data_dir / f"{name}.mat"
        if not path.exists():
            # ищем файл в подпапках (например, datasets/)
            found = sorted(self.data_dir.rglob(f"{name}.mat"))
            if not found:
                raise FileNotFoundError(
                    f"Не найден {name}.mat ни в {self.data_dir}, ни в его подпапках. "
                    f"Положите .mat рядом с DataWork.py или передайте data_dir=..."
                )
            path = found[0]

        mat = loadmat(str(path), simplify_cells=True)
        if name in mat:
            key = name
        else:
            keys = [k_ for k_ in mat if not k_.startswith("__")]
            if len(keys) != 1:
                raise KeyError(f"В {path.name} нет ключа '{name}', есть: {list(mat.keys())}")
            key = keys[0]

        rows, k = [], 0
        for c in mat[key]["cycle"]:
            if c["type"] != "discharge":
                continue
            k += 1
            d = c["data"]
            v = np.atleast_1d(d["Voltage_measured"]).astype(float)
            i = np.atleast_1d(d["Current_measured"]).astype(float)
            t = np.atleast_1d(d["Temperature_measured"]).astype(float)
            tm = np.atleast_1d(d["Time"]).astype(float)

            below = np.where(v < 3.5)[0]
            rows.append({
                "battery": name,
                "cycle": k,
                "voltage": v.mean(),
                "current": i.mean(),
                "temperature": t.mean(),
                "capacity": float(np.atleast_1d(d["Capacity"])[0]),
                # дополнительные признаки
                "ambient_temp": float(c["ambient_temperature"]),
                "v_min": v.min(),
                "v_std": v.std(),
                "t_max": t.max(),
                "t_min": t.min(),
                "duration": tm[-1],
                "time_below_3_5V": tm[below[0]] if len(below) else tm[-1],
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def get_data(self) -> pd.DataFrame:
        """Таблица по всем батареям (кэшируется, возвращается копия)."""
        if self._cache is None:
            frames = [self._load_battery(b) for b in self.batteries]
            df = pd.concat(frames, ignore_index=True)
            df = df.dropna(subset=["voltage", "current", "temperature", "capacity"])
            self._cache = df.reset_index(drop=True)
        return self._cache.copy()


class Datasets(Enum):
    NASA_BATTERY_DATASET = NasaBatteryDataset()


if __name__ == "__main__":
    df = Datasets.NASA_BATTERY_DATASET.value.get_data()
    print(df.groupby("battery").size())
    print(df[["voltage", "current", "temperature", "capacity"]].describe())
