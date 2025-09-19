import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Chargement du dataset
df = pd.read_csv("../../CSV/Encoded/Couleur_OneHot/poker_encoded_couleurOneHot(10000).csv", header=None)

# Séparation des features et des deux sorties (probabilités preflop et fullboard)
X = df.iloc[:, :34]  #34/119 Caractéristiques
y_preflop_prob = df.iloc[:, -2]  # Probabilité Preflop
y_fullboard_prob = df.iloc[:, -1]  # Probabilité Fullboard

# Fonction de transformation probabilité → classe (0 à 4) basée sur des intervalles
def prob_to_class(prob):
    if prob < 0.4:
        return 0
    elif prob < 0.7:
        return 1
    else:
        return 2

# Application de la fonction pour convertir les probabilités en classes
y_class_preflop = y_preflop_prob.apply(prob_to_class)
y_class_fullboard = y_fullboard_prob.apply(prob_to_class)


X_preflop = df.iloc[:, :10]  # 10/34 premières colonnes pour le Preflop
X_fullboard = df.iloc[:, :34]  # 34/119 premières colonnes pour le Fullboard

X_train_preflop, X_temp_preflop, y_train_preflop, y_temp_preflop = train_test_split(
    X_preflop, y_class_preflop, test_size=0.4, random_state=42
)
X_val_preflop, X_test_preflop, y_val_preflop, y_test_preflop = train_test_split(
    X_temp_preflop, y_temp_preflop, test_size=0.5, random_state=42
)

X_train_fullboard, X_temp_fullboard, y_train_fullboard, y_temp_fullboard = train_test_split(
    X_fullboard, y_class_fullboard, test_size=0.4, random_state=42
)
X_val_fullboard, X_test_fullboard, y_val_fullboard, y_test_fullboard = train_test_split(
    X_temp_fullboard, y_temp_fullboard, test_size=0.5, random_state=42
)

def evaluate_model(X_train, X_test, y_train, y_test, label=""):
    # Modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n🎯 Accuracy {label}: {acc:.4f}")
    print(f"\n📊 Rapport de classification {label} :\n", classification_report(y_test, y_pred))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)

    # Trouver les classes uniques dans y_test et y_pred
    unique_labels = np.unique(np.concatenate([y_test, y_pred]))

    # Affichage de la matrice de confusion
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
    disp.plot(cmap="Blues")
    plt.title(f"Matrice de Confusion - {label}")
    plt.show()

    # Courbe ROC
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    y_train_bin = label_binarize(y_train, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]

    classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    classifier.fit(X_train, y_train_bin)
    y_score = classifier.predict_proba(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f"Classe {i} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taux de Faux Positifs (FPR)")
    plt.ylabel("Taux de Vrais Positifs (TPR)")
    plt.title(f"🌈 Courbes ROC - {label}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

# Évaluation pour Preflop
evaluate_model(X_train_preflop, X_test_preflop, y_train_preflop, y_test_preflop, label="Pré-Flop")

# Évaluation pour Fullboard
evaluate_model(X_train_fullboard, X_test_fullboard, y_train_fullboard, y_test_fullboard, label="Fullboard")