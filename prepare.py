"""Calcule une fois les features à partir des données brutes (data/*.txt).

Produit dans artifacts/ :
- features.csv     : les 134 features + cible et niveaux de profile, pour les 2205 cycles
- test_signals.npz : courbes brutes PS2 et FS1 des cycles de test, pour l'affichage

L'interface n'a ainsi besoin que de quelques Mo au lieu des 90 Mo de data/.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from app.features import N_TRAIN, build_features

DATA = Path("data")
OUT = Path("artifacts")


def main():
    ps2 = pd.read_csv(DATA / "PS2.txt", sep="\t", header=None).values
    fs1 = pd.read_csv(DATA / "FS1.txt", sep="\t", header=None).values
    profile = pd.read_csv(DATA / "profile.txt", sep="\t", header=None)
    profile.columns = ["cooler", "valve", "pump_leak", "accumulator", "stable_flag"]

    X = build_features(ps2, fs1)
    X["y"] = (profile["valve"] == 100).astype(int)   # 1 = optimal, 0 = non optimal
    X["valve"] = profile["valve"]
    X["cooler"] = profile["cooler"]
    X.index.name = "cycle"

    OUT.mkdir(exist_ok=True)
    X.to_csv(OUT / "features.csv")
    # Courbe PS2 moyenne par niveau de valve, calculée sur le train uniquement
    valve_tr = profile["valve"].values[:N_TRAIN]
    niveaux = np.array(sorted(profile["valve"].unique(), reverse=True))
    moyennes = np.stack([ps2[:N_TRAIN][valve_tr == v].mean(axis=0) for v in niveaux])

    # float64 conservé : les features recalculées dans l'interface restent identiques
    np.savez_compressed(OUT / "test_signals.npz", ps2=ps2[N_TRAIN:], fs1=fs1[N_TRAIN:],
                        niveaux=niveaux, ps2_moyen_train=moyennes)
    print(f"features.csv : {X.shape} | test_signals.npz : {len(ps2) - N_TRAIN} cycles")


if __name__ == "__main__":
    main()
