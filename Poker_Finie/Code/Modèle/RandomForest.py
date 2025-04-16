import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Chargement du dataset
df = pd.read_csv("../CSV/Encoded/OneHot/poker_encoded_onehot(10000).csv", header=None)  # Pas de header si pas précisé dans le fichier

# Séparation correcte
X_preflop = df.iloc[:, :34]     # Main seulement
X_fullboard = df.iloc[:, :119]   # Main + board

y_preflop = df.iloc[:, 119]      # preflop_win_rate
y_fullboard = df.iloc[:, 120]    # fullboard_win_rate

# Modèle pour preflop
print("🎯 Preflop Win Rate")
mse_list_p = []
r2_list_p = []
mae_list_p = []
explained_variance_p = []
for i in range(10):
    # Séparation du dataset : 60% pour entraînement, 40% pour test et validation
    X_train_p, X_temp_p, y_train_p, y_temp_p = train_test_split(X_preflop, y_preflop, test_size=0.4)

    # Séparation des 40% restants : 50% pour validation, 50% pour test (soit 20% du dataset total)
    X_val_p, X_test_p, y_val_p, y_test_p = train_test_split(X_temp_p, y_temp_p, test_size=0.5)

    # Entraînement du modèle
    model_p = RandomForestRegressor()
    model_p.fit(X_train_p, y_train_p)

    # Prédictions sur le set de test
    y_pred_p = model_p.predict(X_test_p)

    # Calcul des métriques
    mse_list_p.append(mean_squared_error(y_test_p, y_pred_p))
    r2_list_p.append(r2_score(y_test_p, y_pred_p))
    mae_list_p.append(mean_absolute_error(y_test_p, y_pred_p))
    explained_variance_p.append(model_p.score(X_test_p, y_test_p))  # R² sur les données de test

print("MSE moyen :", np.mean(mse_list_p))
print("R² moyen  :", np.mean(r2_list_p))
print("MAE moyen :", np.mean(mae_list_p))
print("Variance expliquée moyenne :", np.mean(explained_variance_p))

# Modèle pour fullboard
print("\n🃏 Fullboard Win Rate")
mse_list_f = []
r2_list_f = []
mae_list_f = []
explained_variance_f = []

for i in range(10):
    # Séparation du dataset : 60% pour entraînement, 40% pour test et validation
    X_train_f, X_temp_f, y_train_f, y_temp_f = train_test_split(X_fullboard, y_fullboard, test_size=0.4)

    # Séparation des 40% restants : 50% pour validation, 50% pour test (soit 20% du dataset total)
    X_val_f, X_test_f, y_val_f, y_test_f = train_test_split(X_temp_f, y_temp_f, test_size=0.5)

    # Entraînement du modèle
    model_f = RandomForestRegressor()
    model_f.fit(X_train_f, y_train_f)

    # Prédictions sur le set de test
    y_pred_f = model_f.predict(X_test_f)

    # Calcul des métriques
    mse_list_f.append(mean_squared_error(y_test_f, y_pred_f))
    r2_list_f.append(r2_score(y_test_f, y_pred_f))
    mae_list_f.append(mean_absolute_error(y_test_f, y_pred_f))
    explained_variance_f.append(model_f.score(X_test_f, y_test_f))  # R² sur les données de test

print("MSE moyen :", np.mean(mse_list_f))
print("R² moyen  :", np.mean(r2_list_f))
print("MAE moyen :", np.mean(mae_list_f))
print("Variance expliquée moyenne :", np.mean(explained_variance_f))