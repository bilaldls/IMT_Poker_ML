import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix
)

# === PARAMETRE SEUIL POUR CONSIDÉRER COMME GAGNANT ===
threshold = 0.75

# Chargement du dataset
df = pd.read_csv("../../CSV/Encoded/OneHot/poker_encoded_onehot(10000).csv", header=None)

# Séparation des features et des labels
X_preflop = df.iloc[:, :34]
X_fullboard = df.iloc[:, :119]

y_preflop = (df.iloc[:, 119] > threshold).astype(int)
y_fullboard = (df.iloc[:, 120] > threshold).astype(int)

# === FONCTION DE CLASSIFICATION NAIVE BAYES AVEC COURBES ===
def evaluate_naive_bayes(X, y, label="Model"):
    print(f"\n🎯 Classification Naive Bayes - {label} Win Rate (> {threshold})")

    acc_list, prec_list, rec_list, f1_list = [], [], [], []
    all_fpr, all_tpr = [], []
    conf_matrices = []

    for i in range(10):
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=i)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=i)

        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc_list.append(accuracy_score(y_test, y_pred))
        prec_list.append(precision_score(y_test, y_pred, zero_division=0))
        rec_list.append(recall_score(y_test, y_pred, zero_division=0))
        f1_list.append(f1_score(y_test, y_pred, zero_division=0))
        conf_matrices.append(confusion_matrix(y_test, y_pred))

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        all_fpr.append(fpr)
        all_tpr.append(tpr)

    print(f"Accuracy moyen : {np.mean(acc_list):.4f}")
    print(f"Precision moyenne : {np.mean(prec_list):.4f}")
    print(f"Recall moyen : {np.mean(rec_list):.4f}")
    print(f"F1-score moyen : {np.mean(f1_list):.4f}")

    # === Courbe ROC moyenne ===
    plt.figure(figsize=(7, 5))
    for fpr, tpr in zip(all_fpr, all_tpr):
        plt.plot(fpr, tpr, color='blue', alpha=0.2)
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(all_fpr, all_tpr)], axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='red', label=f'Moyenne ROC (AUC = {mean_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title(f"Courbe ROC - {label}")
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.legend()
    plt.grid()
    plt.show()

    # === Matrice de confusion moyenne ===
    mean_cm = np.mean(conf_matrices, axis=0)
    plt.figure(figsize=(4, 4))
    sns.heatmap(mean_cm, annot=True, fmt=".1f", cmap="Blues", xticklabels=["Perd", "Gagne"], yticklabels=["Perd", "Gagne"])
    plt.title(f"Matrice de confusion moyenne - {label}")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.show()

# Exécution
evaluate_naive_bayes(X_preflop, y_preflop, label="Preflop")
evaluate_naive_bayes(X_fullboard, y_fullboard, label="Fullboard")