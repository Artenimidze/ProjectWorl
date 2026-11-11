"""
DataWork.py — подготовка датасета Satellite (Satellite Power-System Fault Diagnosis).

Источник: andrewcropper/ilp-datasets (Hugging Face), папка satellite/.
Данные хранятся в виде логических фактов (формат Prolog), поэтому сначала
они превращаются в таблицу: одна строка = один момент времени T.

bk.pl  — телеметрия:  tm001_eod_relay(c_positive, t17).  gmt_mission_phase(eclipse, t17).
exs.pl — примеры:     pos(fault(t17)).  neg(fault(t18)).

Интерфейс для train.py:
    from DataWork import Datasets
    df, feature_names = Datasets.SATELLITE.value.get_data()
df содержит признаки (0/1) и столбцы 'time' (имя шага) и 'label' (1 = неисправность).
"""
import re
from enum import Enum
from pathlib import Path

import pandas as pd

FACT_RE = re.compile(r"^\s*([a-z][A-Za-z0-9_]*)\(([^()]*)\)\s*\.\s*$")
EX_RE = re.compile(r"^\s*(pos|neg)\(\s*fault\(\s*([^()\s,]+)\s*\)\s*\)\s*\.\s*$")
STRUCTURE = {"succ", "eclipse", "solstice", "presolstice", "reconditioning",
             "c_positive", "c_negative"}


class SatelliteDataset:
    def __init__(self, data_dir=None, use_prev=False):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parent
        self.use_prev = use_prev          # добавить показания предыдущего шага (через succ)
        self._cache = None
        self.dropped_no_telemetry = 0

    def _find(self, name: str) -> Path:
        found = sorted(p for p in self.data_dir.rglob(name) if "artifacts" not in p.parts)
        if not found:
            raise FileNotFoundError(
                f"Не найден {name}. Выполните python download_data.py "
                f"(файлы окажутся в data/satellite/)."
            )
        return found[0]

    @staticmethod
    def _split_args(s: str):
        return [a.strip() for a in s.split(",")]

    def get_data(self):
        if self._cache is not None:
            df, feats = self._cache
            return df.copy(), list(feats)

        labels = {}
        for line in self._find("exs.pl").read_text(encoding="utf-8", errors="ignore").splitlines():
            m = EX_RE.match(line)
            if m:
                labels[m.group(2)] = 1 if m.group(1) == "pos" else 0
        if not labels:
            raise ValueError("В exs.pl не найдено ни одного примера вида pos(fault(t)). / neg(fault(t)).")

        channels, phase, prev = {}, {}, {}
        for line in self._find("bk.pl").read_text(encoding="utf-8", errors="ignore").splitlines():
            m = FACT_RE.match(line)
            if not m:
                continue
            rel, args = m.group(1), self._split_args(m.group(2))
            if rel == "succ" and len(args) == 2:
                prev[args[1]] = args[0]
            elif rel == "gmt_mission_phase" and len(args) == 2:
                phase[args[1]] = args[0]
            elif rel.startswith("tm") and len(args) == 2:
                channels.setdefault(rel, {}).setdefault(args[1], args[0])

        # В bk.pl телеметрия есть не для всех шагов из exs.pl (для ~220 отрицательных
        # примеров данных нет). Такие шаги исключаем, а не заполняем «пропуском»:
        # иначе признак «нет данных» выдавал бы класс «норма» (в нём нет ни одной неисправности).
        has_data = lambda t: t in phase or any(t in v for v in channels.values())
        times = [t for t in labels if has_data(t)]
        self.dropped_no_telemetry = len(labels) - len(times)
        raw = pd.DataFrame(index=times)
        raw["phase"] = [phase.get(t, "unknown") for t in times]
        for ch, vals in channels.items():
            raw[ch] = [vals.get(t, "missing") for t in times]
        if self.use_prev:
            cols = list(raw.columns)
            for c in cols:
                lookup = raw[c].to_dict()
                raw[f"{c}_prev"] = [lookup.get(prev.get(t), "missing") for t in times]

        # one-hot: каждый (канал, значение) -> признак 0/1; константные признаки убираем
        X = pd.get_dummies(raw, dtype=float)
        X = X.loc[:, X.nunique() > 1]
        out = X.reset_index(drop=True)
        out.insert(0, "time", times)
        out["label"] = [labels[t] for t in times]
        feature_names = [c for c in out.columns if c not in ("time", "label")]
        self._cache = (out, feature_names)
        return out.copy(), list(feature_names)


class Datasets(Enum):
    SATELLITE = SatelliteDataset()


if __name__ == "__main__":
    df, feats = Datasets.SATELLITE.value.get_data()
    print("Моментов времени:", len(df), "| признаков:", len(feats),
          "| исключено без телеметрии:", Datasets.SATELLITE.value.dropped_no_telemetry)
    print(df["label"].value_counts().rename({0: "норма", 1: "неисправность"}))
