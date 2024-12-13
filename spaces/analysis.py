import streamlit as st
from PIL import Image

st.title('2D Latent Space Analysis')

framework = st.sidebar.selectbox('Framework:', ["AllVSOne", "OneVSAll"])

if framework=="AllVSOne":
    digit = st.sidebar.slider('Anormal', min_value=0, max_value=9, value=0)
    img_path = f'latent_space_anom_{digit}'
else:
    digit = st.sidebar.slider('Normal', min_value=0, max_value=9, value=0)
    img_path = f'latent_space_norm_{digit}'

vae = Image.open(f'{framework}/figures/vae/{img_path}.png')
dsvdd = Image.open(f'{framework}/figures/dsvdd/{img_path}.png')

st.divider()
st.header('VAE 2D Latent Space')
st.image(vae)

st.divider()
st.header('2D T-SNE for The 32D projection of DSVDD')
st.image(dsvdd)