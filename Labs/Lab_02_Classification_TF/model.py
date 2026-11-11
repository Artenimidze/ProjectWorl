from tensorflow.keras import layers, models


def build_model(timesteps: int, input_dim: int):
    """
    LSTM-автоэнкодер: кодирует окно из `timesteps` шагов в вектор и восстанавливает его.
    Обучается на нормальных окнах; на неисправностях ошибка восстановления выше.
    """
    model = models.Sequential([
        layers.Input(shape=(timesteps, input_dim)),
        layers.LSTM(16),                               # кодировщик -> вектор
        layers.RepeatVector(timesteps),
        layers.LSTM(16, return_sequences=True),        # декодировщик
        layers.TimeDistributed(layers.Dense(input_dim)),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model
