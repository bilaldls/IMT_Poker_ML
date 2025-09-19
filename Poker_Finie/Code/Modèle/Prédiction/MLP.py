import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Chargement du dataset
df = pd.read_csv("../../CSV/Encoded/OneHot/poker_encoded_onehot(10000).csv", header=None)

# Séparation des features et labels
X_preflop = df.iloc[:, :34]
X_fullboard = df.iloc[:, :119]

y_preflop = df.iloc[:, 119]
y_fullboard = df.iloc[:, 120]

# Fonction pour créer un modèle MLP générique
def create_mlp_model(input_dim):
    inp = layers.Input(shape=(input_dim,), dtype="float32")
    x = layers.Dense(128, activation="relu")(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="linear")(x)
    model = Model(inp, out)
    model.compile(optimizer="adam", loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

# Fonction d'entraînement et d'évaluation
def train_and_evaluate(X, y, model_name):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_np = X_train.values.astype(np.float32)
    X_val_np = X_val.values.astype(np.float32)
    y_train_np = y_train.values
    y_val_np = y_val.values

    model = create_mlp_model(X.shape[1])

    history = model.fit(
        X_train_np, y_train_np,
        validation_data=(X_val_np, y_val_np),
        batch_size=128,
        epochs=50,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0
    )

    eval_results = model.evaluate(X_val_np, y_val_np, batch_size=128, return_dict=True)
    y_pred = model.predict(X_val_np).flatten()

    mse = mean_squared_error(y_val_np, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val_np, y_pred)
    r2 = r2_score(y_val_np, y_pred)

    print(f"\n📊 Résultats {model_name}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")

    # Courbes
    plt.figure()
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title(f'{model_name} - Courbe de loss')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(history.history['root_mean_squared_error'], label='train RMSE')
    plt.plot(history.history['val_root_mean_squared_error'], label='val RMSE')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.title(f'{model_name} - Courbe de RMSE')
    plt.legend()
    plt.show()

    plt.figure()
    plt.scatter(y_val_np, y_pred, alpha=0.3)
    plt.plot([y_val_np.min(), y_val_np.max()], [y_val_np.min(), y_val_np.max()], 'k--')
    plt.xlabel("Valeurs réelles")
    plt.ylabel("Prédictions")
    plt.title(f'{model_name} - Vrai vs Prédit')
    plt.show()

    return model

# Entraînement du modèle préflop
mlp_preflop = train_and_evaluate(X_preflop, y_preflop, "MLP Préflop")

# Entraînement du modèle fullboard
mlp_fullboard = train_and_evaluate(X_fullboard, y_fullboard, "MLP Fullboard")