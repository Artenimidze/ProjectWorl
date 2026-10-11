from tensorflow.keras import layers, models

def build_model(input_dim=3):
    """
    Регрессия для NASA Battery Dataset.
    Вход: voltage, current, temperature (3 признака).
    Выход: capacity (1 значение).
    """
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()