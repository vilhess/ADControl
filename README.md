# **AnoControl: Controlling the False Anomaly Rate in Anomaly Detection**

Welcome to the official repository for **AnoControl**, a framework designed to control the false anomaly rate in anomaly detection tasks.

---

## **Overview**

This repository provides code implementations for AnoControl applied to anomaly detection problems. Current implementations include the **MNIST dataset**.

---

## **MNIST Dataset**

### **Implemented Features**


We compare the performance of several state-of-the-art algorithms:
- [**Deep One-Class Classification**](http://proceedings.mlr.press/v80/ruff18a/ruff18a.pdf)
- [**F-Anogan**](https://www.sciencedirect.com/science/article/abs/pii/S1361841518302640)
- **Variational Auto-Encoders** (both linear and convolutional)
- [**DROCC**](https://arxiv.org/abs/2002.12718)

---

## **Proposed Frameworks**


###

Two anomaly detection frameworks are implemented and compared:

### **1. One vs All Framework**
This approach is based on the framework used in **Deep One-Class Classification**. The key steps are:
1. **Normal Class**: One digit is designated as the normal class.
2. **Anomalies**: All other digits are treated as anomalies.
3. **Training**: Models are trained on a training set containing only the normal digit.
4. **Testing**: The model is tested on a test set containing both normal digits and anomalies.

This framework is straightforward as the model focuses exclusively on learning the distribution of one normal digit.

---

### **2. All vs One Framework (Proposed)**
Our proposed framework introduces the following:
1. **Anomaly Class**: One digit is designated as the anomaly class.
2. **Normal Class**: The remaining nine digits are treated as the normal class.
3. **Training**: Models are trained on a training set consisting of all normal digits.
4. **Testing**: The model is tested on a test set containing normal digits and anomalies.

This approach is more challenging since the model must learn the combined distribution of nine normal digits, making it a comprehensive evaluation of anomaly detection capabilities.

---

## **Scoring Methods**

Rather than relying on a simple threshold determined from a validation set, we utilize the validation set to construct **marginal p-values**. This approach allows us to control the **false anomaly rate** (FAR) on the test set, ensuring that the probability of falsely detecting a normal sample as an anomaly remains within acceptable bounds.

When comparing models, we evaluate them using two key metrics:
1. **True Anomaly Detection Rate (TADR):** The proportion of anomalies correctly identified by the model.
2. **False Conformity Rate (FCR):** The proportion of normal (conforming) samples incorrectly classified as anomalies.

This dual focus ensures a more balanced and meaningful evaluation of anomaly detection performance.

--- 

### **OneVSAll Folder Structure**

The `OneVSAll` folder contains the implementations for the **One vs All** framework. The folder is organized as follows:

- **`models/`**: Contains the model architectures implemented in PyTorch.
- **`losses/`**: Includes the loss functions used, such as those for Variational Autoencoders (VAEs).
- **`detectors/`**: Implements the training methods and scoring mechanisms for each algorithm. For each digit designated as normal, we simulate training and scoring, resulting in 10 separate training and evaluation processes (one for each digit).
- **`results/`**: Stores the generated figures and calculated p-values for each algorithm and digit.

--- 

### **AllvsOne Folder Structure**

The `AllvsOne` folder contains the implementations for the **All vs One** framework. The folder is organized as follows:

- **`models/`**: Contains the model architectures implemented in PyTorch.
- **`losses/`**: Includes the loss functions used, such as those for Variational Autoencoders (VAEs).
- **`detectors/`**: Implements the training methods and scoring mechanisms for each algorithm. For each digit designated as anormal, we simulate training and scoring, resulting in 10 separate training and evaluation processes (one for each digit).
- **`results/`**: Stores the generated figures and calculated p-values for each algorithm and digit.

### **Analysis file in the root folder**

- **`analysis.py`**: A Streamlit web application to visualize and compare the results across methods. To run the app, execute the following command:
  ```bash
  streamlit run analysis.py
  ```

---

## **Latent space study**

In the folder ```projections```, you will find study on the latent space of both VAE and Deep SVDD. 

---