import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# Chargement du dataset
df = pd.read_csv("../../CSV/Encoded/OneHot/poker_encoded_onehot(10000).csv", header=None)

# Séparation des features et des labels
X_preflop = df.iloc[:, :34]
X_fullboard = df.iloc[:, :119]

# Transformation en 3 classes
def convert_to_multiclass(proba):
    if proba <= 0.4:
        return 0
    elif proba <= 0.7:
        return 1
    else:
        return 2

y_preflop = df.iloc[:, 119].apply(convert_to_multiclass)
y_fullboard = df.iloc[:, 120].apply(convert_to_multiclass)

# === FONCTION DE CLASSIFICATION MULTICLASSE ===
def evaluate_multiclass_logistic_regression(X, y, label="Model", use_smote=False):
    print(f"\n🎯 Classification Multiclasse {label}")

    acc_list, prec_list, rec_list, f1_list = [], [], [], []
    all_conf_matrices = []

    n_classes = 3
    fpr_list = [[] for _ in range(n_classes)]
    tpr_list = [[] for _ in range(n_classes)]
    auc_list = []

    plt.figure(figsize=(10, 8))

    for i in range(5):
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=i)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=i)

        if use_smote:
            smote = SMOTE(random_state=i)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Evaluation
        acc_list.append(accuracy_score(y_test, y_pred))
        prec_list.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
        rec_list.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
        f1_list.append(f1_score(y_test, y_pred, average='macro', zero_division=0))

        cm = confusion_matrix(y_test, y_pred)
        all_conf_matrices.append(cm)

        # === ROC / AUC MULTICLASSE ===
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        for c in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, c], y_proba[:, c])
            fpr_list[c].append(fpr)
            tpr_list[c].append(tpr)
            auc_list.append(auc(fpr, tpr))
            plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Classe {c} (fold {i}) AUC = {auc(fpr, tpr):.2f}')

    # === MOYENNES GLOBALES ===
    print(f"Accuracy moyen : {np.mean(acc_list):.4f}")
    print(f"Precision moyenne (macro) : {np.mean(prec_list):.4f}")
    print(f"Recall moyen (macro) : {np.mean(rec_list):.4f}")
    print(f"F1-score moyen (macro) : {np.mean(f1_list):.4f}")
    print(f"AUC moyenne par classe : {np.mean(auc_list):.4f}")

    # === ROC PLOT FINAL ===
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"Courbes ROC - Multiclasse - {label}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    # === MATRICE DE CONFUSION ===
    mean_cm = np.mean(all_conf_matrices, axis=0).astype(int)
    disp = ConfusionMatrixDisplay(confusion_matrix=mean_cm, display_labels=["Perdant", "Moyen", "Gagnant"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Matrice de Confusion Moyenne - {label}")
    plt.show()

# Exécution
evaluate_multiclass_logistic_regression(X_preflop, y_preflop, label="Preflop")
evaluate_multiclass_logistic_regression(X_fullboard, y_fullboard, label="Fullboard (SMOTE)", use_smote=True)