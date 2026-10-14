"""Extraction des features, reprise à l'identique du notebook (sections 2.6 et 2.7).

Les mêmes fonctions servent à préparer les données (prepare.py) et à prédire
dans l'interface : un cycle est transformé exactement de la même façon.
"""
import numpy as np
import pandas as pd

N_TRAIN = 2000          # consigne : 2000 premiers cycles = train, le reste = test
TAILLE_BLOC = 100       # validation croisée par blocs de 100 cycles consécutifs

# Instant de montée PS2 : seuil 50 bar, fenêtre 8,8 à 10,5 s (indices à 100 Hz)
PS2_DEBUT, PS2_FIN, PS2_SEUIL = 880, 1050, 50
# Instant de montée FS1 : seuil 1 l/min, fenêtre 8,8 à 11 s (indices à 10 Hz)
FS1_DEBUT, FS1_FIN, FS1_SEUIL = 88, 110, 1


def global_stats(df, prefix):
    # Statistiques résumant la forme globale de chaque courbe (une ligne = un cycle)
    return pd.DataFrame({
        f"{prefix}_mean":   df.mean(axis=1),
        f"{prefix}_std":    df.std(axis=1),
        f"{prefix}_max":    df.max(axis=1),
        f"{prefix}_median": df.median(axis=1),
        f"{prefix}_skew":   df.skew(axis=1),
        f"{prefix}_kurt":   df.kurt(axis=1),
    })


def per_second_means(df, freq, prefix):
    # (n_cycles, 60 * freq) -> (n_cycles, 60, freq) -> moyenne sur chaque seconde
    values = df.values.reshape(len(df), -1, freq).mean(axis=2)
    return pd.DataFrame(values, columns=[f"{prefix}_s{i:02d}" for i in range(values.shape[1])])


def t_montee(signal, debut, fin, seuil, freq):
    """Premier instant (en s) où le signal dépasse le seuil dans la fenêtre [debut, fin[."""
    fenetre = np.asarray(signal)[:, debut:fin]
    au_dessus = fenetre > seuil
    if not (fenetre[:, 0] < seuil).all():
        raise ValueError("signal déjà au-dessus du seuil au début de la fenêtre")
    if not au_dessus.any(axis=1).all():
        raise ValueError("aucune montée détectée dans la fenêtre")
    return (debut + au_dessus.argmax(axis=1)) / freq


def build_features(ps2, fs1):
    """ps2 : (n, 6000) à 100 Hz, fs1 : (n, 600) à 10 Hz -> DataFrame (n, 134)."""
    df_ps, df_fs = pd.DataFrame(ps2), pd.DataFrame(fs1)
    X = pd.concat([
        global_stats(df_ps, "PS2"), global_stats(df_fs, "FS1"),
        per_second_means(df_ps, 100, "PS2"), per_second_means(df_fs, 10, "FS1"),
    ], axis=1)
    X["PS2_t_montee"] = t_montee(df_ps.values, PS2_DEBUT, PS2_FIN, PS2_SEUIL, 100)
    X["FS1_t_montee"] = t_montee(df_fs.values, FS1_DEBUT, FS1_FIN, FS1_SEUIL, 10)
    return X
