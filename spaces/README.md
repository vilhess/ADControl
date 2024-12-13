# Latent Space Analysis

This section explores the latent spaces of **Deep SVDD** and **VAEs** (Variational Autoencoders).

---

## **VAE: Variational Autoencoder**

- In this analysis, we focus on a **linear VAE**.  
- The latent space is 2-dimensional, which allows for straightforward visualization and interpretation.  
- Scatter plots and clustering techniques are used to study the structure of the latent space.  

---

## **Deep SVDD: Deep Support Vector Data Description**

- Deep SVDD operates in a latent space of **32 dimensions**.  
- To visualize this high-dimensional space, we use **t-SNE embedding** to reduce the dimensionality to 2.  
- This enables meaningful visualization and analysis of the projected latent space.  

---

## **Analysis**

The detailed analysis is available via a Streamlit web application.  
Run the following command to launch the application:

```bash
streamlit run analysis.py
```