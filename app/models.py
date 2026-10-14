"""Modèles du notebook (section 4.2) et descente de gradient codée à la main."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import N_TRAIN, TAILLE_BLOC

COLONNES_META = ["y", "valve", "cooler"]


def definir_modeles(toutes):
    """Mêmes modèles et hyperparamètres que le notebook. nom -> (colonnes, modèle)."""
    return {
        "Logistique, PS2_t_montee seule": (["PS2_t_montee"], make_pipeline(StandardScaler(), LogisticRegression())),
        "Logistique, PS2_s09 seule":      (["PS2_s09"], make_pipeline(StandardScaler(), LogisticRegression())),
        "KNN, PS2_s09 seule":             (["PS2_s09"], make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))),
        "Logistique, toutes features":    (toutes, make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))),
        "Random Forest, toutes features": (toutes, RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)),
    }


def decouper(df):
    X = df.drop(columns=COLONNES_META)
    return (X.iloc[:N_TRAIN], df["y"].iloc[:N_TRAIN]), (X.iloc[N_TRAIN:], df["y"].iloc[N_TRAIN:])


def entrainer(modeles, X_train, y_train):
    return {nom: modele.fit(X_train[cols], y_train) for nom, (cols, modele) in modeles.items()}


def scores_validation(modeles, X_train, y_train):
    """Validation croisée par blocs de 100 cycles consécutifs (GroupKFold, 5 folds)."""
    groups = np.arange(N_TRAIN) // TAILLE_BLOC
    lignes = {}
    for nom, (cols, modele) in modeles.items():
        res = cross_validate(modele, X_train[cols], y_train, cv=GroupKFold(n_splits=5),
                             groups=groups, scoring=["balanced_accuracy", "roc_auc"])
        lignes[nom] = {
            "BA moyenne": res["test_balanced_accuracy"].mean(), "BA écart-type": res["test_balanced_accuracy"].std(),
            "AUC moyenne": res["test_roc_auc"].mean(), "AUC écart-type": res["test_roc_auc"].std(),
        }
    return pd.DataFrame(lignes).T


def scores_test(entraines, modeles, X_test, y_test):
    lignes = {}
    for nom, modele in entraines.items():
        cols = modeles[nom][0]
        pred = modele.predict(X_test[cols])
        proba = modele.predict_proba(X_test[cols])[:, 1]
        lignes[nom] = {"BA test": balanced_accuracy_score(y_test, pred),
                       "AUC test": roc_auc_score(y_test, proba),
                       "erreurs": int((pred != y_test.values).sum())}
    return pd.DataFrame(lignes).T


def seuil_appris(pipeline):
    """Pour une logistique à une feature : valeur de x où la probabilité vaut 0,5."""
    sc = pipeline.named_steps["standardscaler"]
    lr = pipeline.named_steps["logisticregression"]
    return sc.mean_[0] - lr.intercept_[0] * sc.scale_[0] / lr.coef_[0, 0]


def sigmoide(z):
    return 0.5 * (1 + np.tanh(0.5 * z))   # forme stable numériquement


def descente_gradient(X, y, lr, n_iter, lam, n_images=80):
    """Régression logistique par descente de gradient (batch).

    Perte = log-loss moyenne + lam / 2 * ||w||^2 (le biais n'est pas régularisé).
    Avec lam = 1 / n, on retrouve l'objectif de LogisticRegression(C=1) de scikit-learn.
    Renvoie les itérations enregistrées (pour l'animation) et l'historique complet.
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    w, b = np.zeros(X.shape[1]), 0.0
    a_garder = set(np.unique(np.geomspace(1, n_iter, n_images).astype(int))) | {0}
    images, pertes = [], []
    for it in range(n_iter + 1):
        z = X @ w + b
        perte = np.mean(np.logaddexp(0, z) - y * z) + lam / 2 * w @ w
        pertes.append(perte)
        if it in a_garder:
            images.append({"iter": it, "w": w.copy(), "b": b, "perte": perte,
                           "accuracy": np.mean((z > 0) == y)})
        if it == n_iter:
            break
        ecart = sigmoide(z) - y
        w = w - lr * (X.T @ ecart / len(y) + lam * w)
        b = b - lr * ecart.mean()
    return images, np.array(pertes)
