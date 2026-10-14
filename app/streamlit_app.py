"""Interface : tester les modèles sur les cycles de test et animer la régression logistique.

Lancement : streamlit run app/streamlit_app.py  (après python prepare.py)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression

from features import N_TRAIN, PS2_SEUIL, build_features
from models import (decouper, definir_modeles, descente_gradient, entrainer,
                    scores_test, scores_validation, seuil_appris, sigmoide)

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
COULEURS_VALVE = {100: "#2f9e44", 90: "#f59f00", 80: "#e8590c", 73: "#c92a2a"}
COULEURS_CLASSE = {1: "#2f9e44", 0: "#c92a2a"}

st.set_page_config(page_title="Valve hydraulique : modèles", layout="wide")


# ---------------------------------------------------------------- données et modèles

@st.cache_data
def charger():
    if not (ARTIFACTS / "features.csv").exists():
        st.error("artifacts/ introuvable : lancez d'abord `python prepare.py`.")
        st.stop()
    df = pd.read_csv(ARTIFACTS / "features.csv", index_col="cycle")
    signaux = dict(np.load(ARTIFACTS / "test_signals.npz"))
    return df, signaux


@st.cache_resource
def modeles_entraines():
    df, _ = charger()
    (X_train, y_train), _ = decouper(df)
    modeles = definir_modeles(list(X_train.columns))
    return modeles, entrainer(modeles, X_train, y_train)


@st.cache_data
def tableaux_scores():
    df, _ = charger()
    (X_train, y_train), (X_test, y_test) = decouper(df)
    modeles, entraines = modeles_entraines()
    return scores_validation(modeles, X_train, y_train), scores_test(entraines, modeles, X_test, y_test)


@st.cache_data
def lancer_descente(colonnes, standardiser, lr, n_iter, lam):
    df, _ = charger()
    X = df[list(colonnes)].iloc[:N_TRAIN].values
    y = df["y"].iloc[:N_TRAIN].values
    mu, sd = (X.mean(axis=0), X.std(axis=0)) if standardiser else (np.zeros(X.shape[1]), np.ones(X.shape[1]))
    Z = (X - mu) / sd
    images, pertes = descente_gradient(Z, y, lr, n_iter, lam)
    return Z, y, mu, sd, images, pertes


def controles_animation(images, duree_ms):
    """Boutons Play / Pause et curseur d'itération communs aux deux animations."""
    lecture = dict(frame=dict(duration=duree_ms, redraw=True), fromcurrent=True, transition=dict(duration=0))
    return dict(
        updatemenus=[dict(type="buttons", showactive=False, x=0, y=-0.12, xanchor="left", direction="left",
                          buttons=[dict(label="▶ Play", method="animate", args=[None, lecture]),
                                   dict(label="⏸ Pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(x=0.12, y=-0.05, len=0.88, currentvalue=dict(prefix="itération : "),
                      steps=[dict(method="animate", label=str(img["iter"]),
                                  args=[[str(img["iter"])], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                             for img in images])],
    )


def trace_perte(pertes):
    it = np.arange(1, len(pertes))   # axe log : l'itération 0 n'est pas tracée
    return go.Scatter(x=it, y=pertes[1:], mode="lines", line=dict(color="#adb5bd"), name="perte", showlegend=False)


def marqueur_perte(img):
    return go.Scatter(x=[max(img["iter"], 1)], y=[img["perte"]], mode="markers",
                      marker=dict(size=12, color="#1c7ed6"), showlegend=False)


def titre(img):
    w = ", ".join(f"{v:.2f}" for v in img["w"])
    return f"itération {img['iter']} | w = [{w}] | b = {img['b']:.2f} | perte = {img['perte']:.4f} | accuracy = {img['accuracy']:.3f}"


# ---------------------------------------------------------------- onglet 1 : tester un cycle

def onglet_tester():
    df, signaux = charger()
    modeles, entraines = modeles_entraines()
    n_test = len(signaux["ps2"])

    gauche, droite = st.columns([1, 2])
    with gauche:
        cycle = st.number_input("Cycle de test", min_value=N_TRAIN, max_value=N_TRAIN + n_test - 1, value=N_TRAIN,
                                help="Index du cycle, comme dans le notebook (le test va de 2000 à 2204).")
        decalage = st.slider("Simulation : décaler la montée de pression (ms)", -300, 300, 0, step=10,
                             help="Décale la courbe PS2 dans le temps. Une valve dégradée commute plus tard : "
                                  "décalez vers la droite pour voir à partir de quand les modèles basculent.")
        vue_complete = st.checkbox("Afficher le cycle complet (60 s)", value=False)

    i = cycle - N_TRAIN
    ps2 = np.roll(signaux["ps2"][i], decalage // 10)   # 100 Hz : 1 échantillon = 10 ms
    fs1 = signaux["fs1"][i]
    try:
        feats = build_features(ps2[None, :], fs1[None, :])
    except ValueError as e:
        st.error(f"Features impossibles à calculer pour ce signal : {e}")
        return

    reel = df.loc[cycle]
    with gauche:
        a, b = st.columns(2)
        a.metric("Valve réelle", f"{int(reel['valve'])} %")
        b.metric("Classe réelle", "optimal" if reel["y"] == 1 else "non optimal")
        a.metric("PS2_t_montee", f"{feats['PS2_t_montee'].iloc[0]:.2f} s")
        b.metric("PS2_s09", f"{feats['PS2_s09'].iloc[0]:.2f} bar")
        if decalage == 0:
            identiques = np.allclose(feats.values[0], df.loc[cycle, feats.columns].values.astype(float))
            st.caption("✓ features recalculées depuis le signal brut, identiques à prepare.py" if identiques
                       else "⚠️ features recalculées différentes de prepare.py")

    lignes = []
    for nom, modele in entraines.items():
        cols = modeles[nom][0]
        proba = modele.predict_proba(feats[cols])[0, 1]
        pred = int(proba >= 0.5)
        lignes.append({"modèle": nom, "prédiction": "optimal" if pred else "non optimal",
                       "P(optimal)": proba, "correct": "✓" if pred == reel["y"] else "✗"})
    with gauche:
        st.dataframe(pd.DataFrame(lignes).set_index("modèle"), width="stretch",
                     column_config={"P(optimal)": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f")})
        if decalage != 0:
            st.caption("Le décalage ne touche que PS2 : les modèles « toutes features » voient aussi FS1, inchangé.")

    t = np.arange(len(ps2)) / 100
    fig = go.Figure()
    for v, moyenne in zip(signaux["niveaux"], signaux["ps2_moyen_train"]):
        fig.add_trace(go.Scatter(x=t, y=moyenne, mode="lines", name=f"moyenne train, valve {int(v)} %",
                                 line=dict(color=COULEURS_VALVE[int(v)], dash="dot", width=1.5)))
    fig.add_trace(go.Scatter(x=t, y=ps2, mode="lines", name=f"cycle {cycle}", line=dict(color="#1c7ed6", width=3)))
    fig.add_hline(y=PS2_SEUIL, line=dict(color="grey", dash="dash"), annotation_text="seuil 50 bar")
    fig.add_vline(x=feats["PS2_t_montee"].iloc[0], line=dict(color="#1c7ed6", dash="dash"),
                  annotation_text="t_montee")
    fig.update_layout(height=520, xaxis_title="temps dans le cycle (s)", yaxis_title="pression PS2 (bar)",
                      legend=dict(orientation="h", y=-0.2), title="Courbe PS2 du cycle face aux moyennes du train")
    if not vue_complete:
        fig.update_xaxes(range=[8.8, 10.5])
    with droite:
        st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------- onglet 2 : animer l'entraînement

def figure_1d(Z, y, mu, sd, images, pertes, valve, ref, duree_ms):
    t = Z[:, 0] * sd[0] + mu[0]   # affichage en secondes
    xmin, xmax = t.min() - 0.03, t.max() + 0.03
    grille = np.linspace(xmin, xmax, 300)

    def courbe(w, b):
        return sigmoide(w * (grille - mu[0]) / sd[0] + b)

    def seuil(img):
        w, b = img["w"][0], img["b"]
        x = mu[0] - b * sd[0] / w if w != 0 else None
        return go.Scatter(x=[x, x], y=[-0.1, 1.1], mode="lines", line=dict(color="#1c7ed6", dash="dash"),
                          name="seuil (p = 0,5)")

    def sig(img):
        return go.Scatter(x=grille, y=courbe(img["w"][0], img["b"]), mode="lines",
                          line=dict(color="#1c7ed6", width=3), name="descente de gradient")

    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38],
                        subplot_titles=("P(optimal) en fonction de PS2_t_montee", "Perte (axe log)"))
    bruit = np.random.default_rng(0).uniform(-0.04, 0.04, len(y))   # sépare visuellement les points
    for v, couleur in COULEURS_VALVE.items():
        m = valve == v
        fig.add_trace(go.Scatter(x=t[m], y=y[m] + bruit[m], mode="markers", name=f"valve {v} %",
                                 marker=dict(color=couleur, size=5, opacity=0.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=grille, y=courbe(*ref), mode="lines", name="scikit-learn (C = 1)",
                             line=dict(color="black", dash="dot")), row=1, col=1)
    premiere = images[0]
    fig.add_trace(sig(premiere), row=1, col=1)       # trace 5
    fig.add_trace(seuil(premiere), row=1, col=1)     # trace 6
    fig.add_trace(trace_perte(pertes), row=1, col=2)
    fig.add_trace(marqueur_perte(premiere), row=1, col=2)   # trace 8

    fig.frames = [go.Frame(name=str(img["iter"]), data=[sig(img), seuil(img), marqueur_perte(img)],
                           traces=[5, 6, 8], layout=dict(title_text=titre(img))) for img in images]
    fig.update_xaxes(range=[xmin, xmax], title_text="PS2_t_montee (s)", row=1, col=1)
    fig.update_yaxes(range=[-0.1, 1.1], title_text="y (0 = non optimal, 1 = optimal)", row=1, col=1)
    fig.update_xaxes(type="log", title_text="itération", row=1, col=2)
    fig.update_layout(height=560, title_text=titre(premiere), legend=dict(orientation="h", y=-0.3),
                      margin=dict(b=170), **controles_animation(images, duree_ms))
    return fig


def figure_2d(Z, y, images, pertes, colonnes, standardiser, duree_ms):
    marge = 0.1 * (Z.max(axis=0) - Z.min(axis=0))
    (x0, y0), (x1, y1) = Z.min(axis=0) - marge, Z.max(axis=0) + marge
    gx, gy = np.linspace(x0, x1, 60), np.linspace(y0, y1, 60)
    GX, GY = np.meshgrid(gx, gy)

    def fond(img):
        p = sigmoide(img["w"][0] * GX + img["w"][1] * GY + img["b"])
        return go.Heatmap(x=gx, y=gy, z=p, zmin=0, zmax=1, colorscale="RdYlGn", opacity=0.35,
                          showscale=False, hoverinfo="skip")

    def frontiere(img):
        (w1, w2), b = img["w"], img["b"]
        if w1 == 0 and w2 == 0:
            xs, ys = [None], [None]
        elif abs(w2) >= abs(w1):
            xs = [x0, x1]
            ys = [-(w1 * x + b) / w2 for x in xs]
        else:
            ys = [y0, y1]
            xs = [-(w2 * v + b) / w1 for v in ys]
        return go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="black", width=3), name="frontière (p = 0,5)")

    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38],
                        subplot_titles=("Frontière de décision", "Perte (axe log)"))
    premiere = images[0]
    fig.add_trace(fond(premiere), row=1, col=1)       # trace 0
    for classe, nom in [(0, "non optimal"), (1, "optimal")]:
        m = y == classe
        fig.add_trace(go.Scatter(x=Z[m, 0], y=Z[m, 1], mode="markers", name=nom,
                                 marker=dict(color=COULEURS_CLASSE[classe], size=5, opacity=0.6,
                                             line=dict(color="white", width=0.5))), row=1, col=1)
    fig.add_trace(frontiere(premiere), row=1, col=1)  # trace 3
    fig.add_trace(trace_perte(pertes), row=1, col=2)
    fig.add_trace(marqueur_perte(premiere), row=1, col=2)   # trace 5

    fig.frames = [go.Frame(name=str(img["iter"]), data=[fond(img), frontiere(img), marqueur_perte(img)],
                           traces=[0, 3, 5], layout=dict(title_text=titre(img))) for img in images]
    suffixe = " (standardisé)" if standardiser else ""
    fig.update_xaxes(range=[x0, x1], title_text=colonnes[0] + suffixe, row=1, col=1)
    fig.update_yaxes(range=[y0, y1], title_text=colonnes[1] + suffixe, row=1, col=1)
    fig.update_xaxes(type="log", title_text="itération", row=1, col=2)
    fig.update_layout(height=600, title_text=titre(premiere), legend=dict(orientation="h", y=-0.3),
                      margin=dict(b=170), **controles_animation(images, duree_ms))
    return fig


def onglet_animer():
    df, _ = charger()
    st.markdown(
        "La régression logistique est entraînée **à la main** par descente de gradient sur les 2000 cycles du train : "
        "à chaque itération, `w ← w − lr · ∇perte`. Appuyez sur **Play** pour voir la sigmoïde (ou la frontière) se former."
    )
    mode = st.radio("Animation", ["1 feature : PS2_t_montee", "2 features : PS2_t_montee + PS2_s09"], horizontal=True)
    deux_d = mode.startswith("2")

    c1, c2, c3, c4, c5 = st.columns(5)
    standardiser = c1.checkbox("Standardiser les features", value=True, disabled=not deux_d,
                               help="Décochez pour voir la descente de gradient ramer sur des échelles très différentes "
                                    "(t_montee ≈ 9,5 s, PS2_s09 ≈ 78 bar). Il faut alors un learning rate minuscule.")
    standardiser = standardiser or not deux_d
    lrs = [1e-5, 1e-4, 5e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0]
    lr = c2.select_slider("Learning rate", options=lrs, value=1.0 if standardiser else 5e-4)
    n_iter = c3.select_slider("Itérations", options=[50, 100, 200, 500, 1000, 2000, 5000], value=500)
    lam_options = {"0 (aucune)": 0.0, "1/n (= C = 1 de sklearn)": 1 / N_TRAIN, "0,01": 0.01, "0,1": 0.1}
    lam = lam_options[c4.selectbox("Régularisation L2 λ", list(lam_options), index=1)]
    duree_ms = c5.select_slider("Vitesse (ms par image)", options=[30, 60, 100, 200, 400], value=100)

    colonnes = ("PS2_t_montee", "PS2_s09") if deux_d else ("PS2_t_montee",)
    Z, y, mu, sd, images, pertes = lancer_descente(colonnes, standardiser, lr, n_iter, lam)
    if not np.isfinite(pertes).all():
        st.error("La perte diverge : learning rate trop grand pour ces données. Baissez-le.")
        return
    if pertes[-1] > pertes[0]:
        st.warning(f"La perte finale ({pertes[-1]:.2f}) dépasse la perte de départ ({pertes[0]:.2f}) : "
                   "le learning rate est trop grand, la descente oscille au lieu de converger.")

    if deux_d:
        st.plotly_chart(figure_2d(Z, y, images, pertes, colonnes, standardiser, duree_ms), width="stretch")
    else:
        # Référence : solution de scikit-learn sur les mêmes données standardisées
        ref = LogisticRegression().fit(Z, y)
        valve = df["valve"].iloc[:N_TRAIN].values
        st.plotly_chart(figure_1d(Z, y, mu, sd, images, pertes, valve, (ref.coef_[0, 0], ref.intercept_[0]), duree_ms),
                        width="stretch")
        st.caption(f"scikit-learn (C = 1) : w = {ref.coef_[0, 0]:.3f}, b = {ref.intercept_[0]:.3f}. "
                   "Avec λ = 1/n et assez d'itérations, la descente de gradient converge vers la même solution.")

    with st.expander("Pourquoi w grandit sans fin quand λ = 0 ?"):
        st.markdown(
            "Les classes du train sont **parfaitement séparables**. Sans régularisation, la perte peut toujours baisser "
            "en rendant la sigmoïde plus raide : `w` grandit indéfiniment et la sigmoïde tend vers une marche. "
            "La pénalité `λ/2 · ||w||²` stoppe cette course et fixe une solution finie."
        )


# ---------------------------------------------------------------- onglet 3 : comparer

def onglet_comparer():
    modeles, entraines = modeles_entraines()
    with st.spinner("Validation croisée par blocs en cours…"):
        cv, test = tableaux_scores()
    st.subheader("Validation croisée par blocs (train, 5 folds, blocs de 100 cycles)")
    st.dataframe(cv.style.format("{:.3f}"), width="stretch")
    st.subheader("Test final (205 cycles, entraînement sur les 2000 du train)")
    st.dataframe(test.style.format({"BA test": "{:.3f}", "AUC test": "{:.3f}"}), width="stretch")
    st.subheader("Seuils appris par les logistiques à une feature")
    st.markdown(f"- `PS2_t_montee` : prédit optimal si **t_montee < {seuil_appris(entraines['Logistique, PS2_t_montee seule']):.3f} s**\n"
                f"- `PS2_s09` : prédit optimal si **PS2_s09 > {seuil_appris(entraines['Logistique, PS2_s09 seule']):.2f} bar**")


# ---------------------------------------------------------------- page

st.title("Maintenance prédictive d'une valve hydraulique")
st.caption("Prédire si la valve est optimale (= 100 %) à partir de la pression PS2 et du débit FS1 d'un cycle de 60 s.")
tab1, tab2, tab3 = st.tabs(["🔍 Tester un cycle", "🎬 Animer l'entraînement", "📊 Comparer les modèles"])
with tab1:
    onglet_tester()
with tab2:
    onglet_animer()
with tab3:
    onglet_comparer()
