import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Chargement du dataset
df = pd.read_csv("../../CSV/Encoded/OneHot/poker_encoded_onehot(10000).csv", header=None)

# Séparation correcte
X_preflop = df.iloc[:, :34]     # Main seulement
X_fullboard = df.iloc[:, :119]   # Main + board

y_preflop = df.iloc[:, 119]      # preflop_win_rate
y_fullboard = df.iloc[:, 120]    # fullboard_win_rate

# Choix du modèle : LinearRegression, Ridge, ou Lasso
ModelClass = LinearRegression     # Change ici : Ridge / Lasso si tu veux tester autre chose
model_params = {}                # Exemples : {'alpha': 1.0} pour Ridge ou Lasso

# Modèle pour preflop
print("🎯 Preflop Win Rate")
mse_list_p, r2_list_p, mae_list_p, explained_variance_p = [], [], [], []
train_errors_p, test_errors_p = [], []  # Pour courbe d'apprentissage

for i in range(10):
    X_train_p, X_temp_p, y_train_p, y_temp_p = train_test_split(X_preflop, y_preflop, test_size=0.4)
    X_val_p, X_test_p, y_val_p, y_test_p = train_test_split(X_temp_p, y_temp_p, test_size=0.5)

    model_p = ModelClass(**model_params)
    model_p.fit(X_train_p, y_train_p)
    y_pred_p = model_p.predict(X_test_p)

    mse_list_p.append(mean_squared_error(y_test_p, y_pred_p))
    r2_list_p.append(r2_score(y_test_p, y_pred_p))
    mae_list_p.append(mean_absolute_error(y_test_p, y_pred_p))
    explained_variance_p.append(model_p.score(X_test_p, y_test_p))

    # Erreurs pour la courbe d'apprentissage
    train_errors_p.append(mean_squared_error(y_train_p, model_p.predict(X_train_p)))
    test_errors_p.append(mean_squared_error(y_test_p, y_pred_p))

print("MSE moyen :", np.mean(mse_list_p))
print("R² moyen  :", np.mean(r2_list_p))
print("MAE moyen :", np.mean(mae_list_p))
print("Variance expliquée moyenne :", np.mean(explained_variance_p))

# Courbe d'apprentissage pour le modèle préflop
plt.figure(figsize=(10,6))
plt.plot(range(1, 11), train_errors_p, label="Train Error")
plt.plot(range(1, 11), test_errors_p, label="Test Error")
plt.xlabel('Iterations')
plt.ylabel('Erreur quadratique moyenne (MSE)')
plt.title('Courbe d\'Apprentissage - Modèle Preflop')
plt.legend()
plt.show()

# Courbe de régression (Prévisions vs Réelles)
plt.figure(figsize=(10,6))
plt.scatter(y_test_p, y_pred_p, color='blue')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Ligne d'identité
plt.xlabel('Valeurs réelles')
plt.ylabel('Valeurs prédites')
plt.title('Prévisions vs Réelles - Modèle Preflop')
plt.show()

# Modèle pour fullboard
print("\n🃏 Fullboard Win Rate")
mse_list_f, r2_list_f, mae_list_f, explained_variance_f = [], [], [], []
train_errors_f, test_errors_f = [], []  # Pour courbe d'apprentissage

for i in range(10):
    X_train_f, X_temp_f, y_train_f, y_temp_f = train_test_split(X_fullboard, y_fullboard, test_size=0.4)
    X_val_f, X_test_f, y_val_f, y_test_f = train_test_split(X_temp_f, y_temp_f, test_size=0.5)

    model_f = ModelClass(**model_params)
    model_f.fit(X_train_f, y_train_f)
    y_pred_f = model_f.predict(X_test_f)

    mse_list_f.append(mean_squared_error(y_test_f, y_pred_f))
    r2_list_f.append(r2_score(y_test_f, y_pred_f))
    mae_list_f.append(mean_absolute_error(y_test_f, y_pred_f))
    explained_variance_f.append(model_f.score(X_test_f, y_test_f))

    # Erreurs pour la courbe d'apprentissage
    train_errors_f.append(mean_squared_error(y_train_f, model_f.predict(X_train_f)))
    test_errors_f.append(mean_squared_error(y_test_f, y_pred_f))

print("MSE moyen :", np.mean(mse_list_f))
print("R² moyen  :", np.mean(r2_list_f))
print("MAE moyen :", np.mean(mae_list_f))
print("Variance expliquée moyenne :", np.mean(explained_variance_f))

# Courbe d'apprentissage pour le modèle fullboard
plt.figure(figsize=(10,6))
plt.plot(range(1, 11), train_errors_f, label="Train Error")
plt.plot(range(1, 11), test_errors_f, label="Test Error")
plt.xlabel('Iterations')
plt.ylabel('Erreur quadratique moyenne (MSE)')
plt.title('Courbe d\'Apprentissage - Modèle Fullboard')
plt.legend()
plt.show()

# Courbe de régression (Prévisions vs Réelles) pour fullboard
plt.figure(figsize=(10,6))
plt.scatter(y_test_f, y_pred_f, color='green')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Ligne d'identité
plt.xlabel('Valeurs réelles')
plt.ylabel('Valeurs prédites')
plt.title('Prévisions vs Réelles - Modèle Fullboard')
plt.show()