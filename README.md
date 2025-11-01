# 🤖 Portfolio Project: Customer Churn Analysis with Neural Networks

## 🎯 Objective

This project was developed as part of my Data Science portfolio. The objective is to build and evaluate a Deep Learning model (using TensorFlow/Keras) capable of predicting which customers of a telecommunications company (`Telco`) are most likely to cancel their services (*churn*).

The primary focus is not just on overall accuracy, but on developing a model that is **useful for the business**—meaning it is effective at **identifying at-risk customers** so the company can take retention actions.

## 📊 Data Source

The dataset used is the "Telco Customer Churn," a popular public dataset available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

It contains customer demographic information, services they have signed up for, contract and billing details, and the target column: `Churn` (Yes/No).

## 🛠️ Methodology and Process

The project was divided into the following stages:

### 1. Data Cleaning and Preparation
* Handled missing values (the `TotalCharges` column was corrected and filled).
* Converted binary text data (like 'Yes'/'No') to numerical format (1/0).
* Applied **One-Hot Encoding** (via `pd.get_dummies`) to categorical columns (like `InternetService` and `Contract`) to prepare the data for the AI model.
* The `customerID` column was dropped as it has no predictive value.

### 2. Feature Scaling
* The data was split using `train_test_split` (80% for training, 20% for testing), using `stratify=y` to ensure the churn proportion was identical in both sets.
* The **`StandardScaler`** from `sklearn` was applied to the training and testing data, a crucial step for the performance of neural networks.

### 3. Model Building and Training (TensorFlow/Keras)
* A `Sequential` Neural Network was built with the following architecture:
    * Input Layer (100 neurons, `relu`, with the correct `input_shape`)
    * Hidden Layer (50 neurons, `relu`)
    * `Dropout` Layer (0.25) to prevent *overfitting*.
    * Output Layer (1 neuron, `sigmoid`) for binary classification.
* The model was compiled with `optimizer='adam'` and `loss='binary_crossentropy'`.

### 4. Experimentation and Evaluation

To find the best model, three main experiments were conducted:

1.  **Model 1 (Baseline):** Trained with `EarlyStopping` to find the best performance before *overfitting* occurred.
2.  **Model 2 (Overfit):** Trained for 200 epochs *without* `EarlyStopping`, to visually prove the effect of overfitting (where the test error begins to rise).
3.  **Model 3 (Business-Optimized):** The dataset is imbalanced (far more "No Churn" than "Churn"). Model 1 was re-trained using **`class_weight`** to more heavily penalize the model for misclassifying "Churn" cases, forcing it to improve `recall`.

## 📈 Results and Conclusion

The final evaluation proved that accuracy is not the best metric for this business problem.

* **Model 1 (Baseline)** achieved ~79% accuracy but had a weak `recall` (54%), **missing 171 customers** who were about to churn.
* **Model 3 (Optimized with `class_weight`)** had a *lower* overall accuracy (~73%) but was **infinitely more useful**:
    * Its **Recall** (ability to *find* churning customers) jumped from 54% to **76%**.
    * The model **only missed 88 customers**, saving 83 clients that Model 1 would have lost.

### Confusion Matrix Comparison

(This is where you should paste the side-by-side image of your Model 1 (blue) and Model 3 (green) confusion matrices!)
