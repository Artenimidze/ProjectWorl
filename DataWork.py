"""
:Date: 2026-10-10
:Version: 1
:Authors: 
- Artenimidze\n
Модуль для работы с датасетами
"""
from enum import Enum
import pandas as pd
from scipy.io import loadmat
from pathlib import Path
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler

file = Path(__file__).resolve().parent

logger = logging.getLogger("dataset")
logging.basicConfig(level=logging.INFO)

class Nasa_Battery_Dataset:
    """Датасет Nasa Battery Dataset"""
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.discharge_data = []
        if path is None:
            self.path = file / "datasets/Nasa Battery Dataset"
        if not self.path.exists():
            raise FileNotFoundError("Папка не найдена!")
        self._load_data()

    def _load_data(self) -> None:
        for matfile in self.path.glob("*.mat"):
            battery_name = matfile.stem
            mat = loadmat(matfile, simplify_cells=True)
            if battery_name not in mat:
                logger.warning(f"{matfile.name}: ключ {battery_name} не найден, пропуск")
                continue

            cycles = mat[battery_name]["cycle"]
            discharge_cycles = []
            cycle_num = 0
            for cycle in cycles:
                if cycle['type'] != 'discharge':
                    continue

                cycle_num += 1
                data = cycle['data']
                discharge_cycles.append({
                    'cycle': cycle_num,
                    'voltage': np.mean(data['Voltage_measured']),
                    'current': np.mean(data['Current_measured']),
                    'temperature': np.mean(data['Temperature_measured']),
                    'capacity': float(data['Capacity']),
                })
            logger.info(f"{matfile} обработан")
            df = pd.DataFrame(discharge_cycles)
            self.discharge_data.append({
                'battery_id': battery_name,
                'data':df
            })

    def get_battery(self, battery_id) -> dict[str, float]:
        for entry in self.discharge_data:
            if entry["battery_id"] == battery_id:
                return entry["data"]

    def get_data(self) -> pd.DataFrame:
        frames = []
        for entry in self.discharge_data:
            df = entry["data"].copy()
            df["battery"] = entry["battery_id"]
            frames.append(df)
        return pd.concat(frames, ignore_index=True)


class Datasets(Enum):
    """
    Датасеты, используемы в проекте\n
    \n
    Датасет *NASA Battery Dataset* уже имеет разделение на обучающие и тестовые данные!
    """
    
    NASA_BATTERY_DATASET = Nasa_Battery_Dataset()
    #SATELLITE_POWER_SYSTEM_FAULT_DIAGNOSIS =
    #EUROSAT =
    #NASA_SMAP =
    #NASA_SMAP_AND_MSL =

class DataSet():
    def __init__(self, dataset_type: Datasets, normalize: bool = True, test_split_ratio: float = 0.3) -> None:
        """
        Работа с датасетами: обработка, нормализация, разделение на обучающую и тестовую выборки
        
        Параметры
        ---------
        **dataset_type**: тип используемого датасета. Датасет нужно указывать из :class:`Datasets`\n
        **normalize**: нормализация данных через :meth:`normalize()`. По умолчанию True\n
        **test_split**: соотношение тестовой выборки к обучающей. По умолчанию 0.3. **Внимание**: некоторые датасеты уже имеют как тестовые, 
                так и обучающие данные, поэтому к ним данный параметр не применяется!\n
        """
        self.X_train = None
        self.X_test = None
        self.y_test = None
        self.y_train = None
        if dataset_type is None:
            raise ValueError("dataset_type не указан!")
        if not isinstance(dataset_type, Datasets):
            raise ValueError("dataset_type указан неверно! Используйте датасет из Datasets")
        self.dataset = dataset_type
        self.test_split(test_split_ratio)
        if normalize:
            self.normalize()


    def normalize(self) -> None:
        """
        Нормализует данные
        """
        if self.dataset is None:
            raise ValueError("Датасет не указан!")
        if self.dataset is Datasets.NASA_BATTERY_DATASET:
            # MinMaxScaler обучается ТОЛЬКО на тренировочных данных, чтобы избежать утечки
            scaler_X = MinMaxScaler(feature_range=(0, 1))
            scaler_y = MinMaxScaler(feature_range=(0, 1))

            self.X_train = scaler_X.fit_transform(self.X_train)
            self.X_test = scaler_X.transform(self.X_train)

            # Нормализуем целевую переменную (ёмкость) тоже, если это нужно для обучения сети
            self.y_train = scaler_y.fit_transform(self.y_train.reshape(-1, 1)).ravel()
            self.y_test = scaler_y.transform(self.y_test.reshape(-1, 1)).ravel()

    def test_split(self, ratio: float = 0.3) -> None:
        """
        Делит данные на тестовую и обучающую
        """
        if self.dataset is None:
            raise ValueError("Датасет не указан!")
        if self.dataset is Datasets.NASA_BATTERY_DATASET:
            # Для этого датасета необходимо разделение LOBO (Leave One Battery Out)
            df_all = self.dataset.value.get_data()
            feature_cols = ['voltage', 'current', 'temperature']
            target_col = 'capacity'

            # Разделение на train (B0005, B0006, B0007) и test (B0018)
            train_mask = df_all['battery'].isin(['B0005', 'B0006', 'B0007'])
            test_mask = df_all['battery'] == 'B0018'

            self.X_train = df_all.loc[train_mask, feature_cols].values
            self.y_train = df_all.loc[train_mask, target_col].values

            self.X_test = df_all.loc[test_mask, feature_cols].values
            self.y_test = df_all.loc[test_mask, target_col].values

    def get_dataset(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.X_train is None:
            self.test_split()
        return (self.X_train, self.X_test, self.y_train, self.y_test)

    @classmethod
    def create(cls, dataset_type: Datasets, normalize: bool = True, test_split_ratio: float = 0.3) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Создаёт датасет

        Параметры
        ---------
        **dataset_type**: тип используемого датасета. Датасет нужно указывать из :class:`Datasets`\n
        **normalize**: нормализация данных через :method:`Dataset.normalize()`. По умолчанию True\n
        **test_split**: соотношение тестовой выборки к обучающей. По умолчанию 0.3. **Внимание**: некоторые датасеты уже имеют как тестовые, 
        так и обучающие данные, поэтому к ним данный параметр не применяется!\n
        Возвращает
        ----------
        X_train, X_test, y_train, y_test\n
        То есть **X_train, X_test, y_train, y_test = Dataset.create()**
        """
        data = cls(dataset_type=dataset_type, normalize=normalize, test_split_ratio=test_split_ratio)
        return data.get_dataset()

# Пример
if __name__ == "__main__":
    print(DataSet.create(Datasets.NASA_BATTERY_DATASET))