import streamlit as st
from PIL import Image
import json
import pandas as pd
import numpy as np
import json

# Configurer la page
st.set_page_config(layout="wide")
st.title("Deep Models for Anomaly Detection on MNIST")

# Sidebar : Paramètres globaux
dataset = st.sidebar.selectbox("Dataset :", ["MNIST"])
pb = st.sidebar.selectbox("Problem :", ["OneVSAll", "AllVSOne"])
threshold = st.sidebar.slider("Threshold:", min_value=0.0, max_value=0.2, step=0.01, value=0.05)

def display_images(normal_digit, pb):
    """Affiche les images pour chaque modèle."""
    images = {
        "Deep One Class": f"{pb}/results/figures/deepsvdd/mean_scores_{normal_digit}.jpg",
        "Linear VAE": f"{pb}/results/figures/vae/mean_scores_{normal_digit}.jpg",
        "Conv VAE": f"{pb}/results/figures/cvae/mean_scores_{normal_digit}.jpg",
        "f-ANOGAN": f"{pb}/results/figures/fanogan/mean_scores_{normal_digit}.jpg",
        "DROCC": f"{pb}/results/figures/drocc/mean_scores_{normal_digit}.jpg",
    }
    cols = st.columns(2)
    for i, (title, path) in enumerate(images.items()):
        with cols[i % 2]:
            st.header(title)
            st.image(Image.open(path))
    st.divider()

def load_p_values(pb, digit):
    """Charge les p-values pour chaque modèle."""
    paths = {
        "Deep SVDD": f"{pb}/results/p_values/deepsvdd/pval_{digit}.json",
        "VAE": f"{pb}/results/p_values/vae/pval_{digit}.json",
        "CVAE": f"{pb}/results/p_values/cvae/pval_{digit}.json",
        "f-ANOGAN": f"{pb}/results/p_values/fanogan/pval_{digit}.json",
        "DROCC": f"{pb}/results/p_values/drocc/pval_{digit}.json",
    }
    return {model: json.load(open(path)) for model, path in paths.items()}

def compute_results(p_values, threshold, normal_digit, condition_label):
    """Calcule les résultats des tests statistiques."""
    results = []
    for digit in range(10):
        digit_results = {"Digit": digit, condition_label: "Yes" if digit == normal_digit else "No", "Threshold": threshold}
        for model, values in p_values.items():
            p_vals, test_size = values[str(digit)]
            p_vals = np.asarray(p_vals)
            n_rejected = (p_vals < threshold).sum().item()
            rejection_rate = n_rejected / test_size

            digit_results[f"Rejections {model}"] = f"{n_rejected}/{test_size}"
            digit_results[f"Rejection Rate {model}"] = f"{rejection_rate:.3%}"

        results.append(digit_results)
    return pd.DataFrame(results)

def load_auc(problem="OneVSAll"):

    with open(f'{problem}/results/roc_auc.json', "r") as f:
        results = json.load(f)

    df = pd.DataFrame(results)
    df.index = range(10)
    if problem=="OneVSAll":
        df.index.name="Normal digit"
    else:
        df.index.name="Anormal digit"
    pd.set_option('display.float_format', '{:.3f}'.format)
    return df

# Condition : OneVSAll ou AllVSOne
if pb == "OneVSAll":
    st.subheader("One class of digit is considered as the normal class")
    st.sidebar.title("Parameters:")
    normal_digit = st.sidebar.slider("Normal digit:", min_value=0, max_value=9, value=0, step=1)

    auc = load_auc(pb)
    st.header("ROC-AUC Scores (for all digits)")
    st.table(auc)
    st.divider()

    display_images(normal_digit, pb)
    st.header("Statistical Tests")
    p_values = load_p_values(pb, normal_digit)
    results_df = compute_results(p_values, threshold, normal_digit, "Normal")
    st.table(results_df)

else:
    st.subheader("One class of digit is considered as the anomaly class")
    st.sidebar.title("Parameters:")
    anormal_digit = st.sidebar.slider("Anormal digit:", min_value=0, max_value=9, value=0, step=1)

    auc = load_auc(pb)
    st.header("ROC-AUC Scores (for all digits)")
    st.table(auc)
    st.divider()

    display_images(anormal_digit, pb)
    st.header("Statistical Tests")
    p_values = load_p_values(pb, anormal_digit)
    results_df = compute_results(p_values, threshold, anormal_digit, "Anormal")
    st.table(results_df)