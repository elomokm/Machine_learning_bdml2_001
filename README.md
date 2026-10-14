# Maintenance prédictive d'une valve hydraulique : interface

Appli Streamlit en 3 onglets :
- **Tester un cycle** : prédiction des 5 modèles du notebook sur un cycle de test, avec simulation d'un retard de commutation.
- **Animer l'entraînement** : régression logistique par descente de gradient, en 1D (sigmoïde) ou 2D (frontière de décision).
- **Comparer les modèles** : validation croisée par blocs et score sur le test.

## Données

`data/` n'est pas versionné. Y placer `PS2.txt`, `FS1.txt` et `profile.txt` (dataset UCI *Condition monitoring of hydraulic systems*) avant de lancer `prepare.py` ou `docker build`.

## Lancer en local

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python prepare.py                       # data/ -> artifacts/ (2 s)
.venv/bin/streamlit run app/streamlit_app.py      # http://localhost:8501
```

## Lancer avec Docker

```bash
docker build -t valve-app .
docker run --rm -p 8501:8501 valve-app            # http://localhost:8501
```

Le build calcule les features à partir de `data/` dans une étape intermédiaire : l'image finale ne contient que le code et `artifacts/`.

## Fichiers

| Fichier | Rôle |
|---|---|
| `app/features.py` | extraction des features, identique au notebook (sections 2.6 et 2.7) |
| `app/models.py` | modèles du notebook (section 4.2) et descente de gradient codée à la main |
| `app/streamlit_app.py` | l'interface |
| `prepare.py` | calcule `artifacts/features.csv` et `artifacts/test_signals.npz` |
