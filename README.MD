# 🚗 Car Price Prediction: From Scratch to Production

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-Linear_Algebra-deepgreen)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-Data_Preprocessing-orange)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Regression-red)

## 📌 Project Overview
This project is an end-to-end Machine Learning pipeline designed to predict vehicle market prices based on historical specifications. 

**What makes this project stand out?** Instead of relying entirely on black-box high-level APIs like Scikit-Learn's estimators, the core regression algorithms—including the **Analytical Normal Equation** and **Ridge Regression ($L_2$ Regularization)**—are built entirely **from scratch** using pure Linear Algebra (NumPy). This demonstrates a deep understanding of the underlying mathematics, matrix operations, and statistical behaviors of machine learning models.

## ✨ Key Features & Technical Depth
* **Mathematical Implementation:** Built the Normal Equation `w = (X^T X)^-1 X^T y` and Ridge Penalty `w = (X^T X + rI)^-1 X^T y` entirely from scratch.
* **Target Normalization:** Applied Logarithmic Transformation (`np.log`) to the highly skewed `price` variable to achieve a normal distribution, improving linear model stability.
* **Advanced Feature Engineering:** Extracted mathematical correlations by engineering a vehicle `age` feature, which slashed the initial error by over 30%.
* **Robust Preprocessing Pipeline:** Utilized Scikit-Learn's `ColumnTransformer` and `OneHotEncoder` to dynamically process high-cardinality categorical features.
* **Overcoming Matrix Singularity:** Diagnosed and mathematically resolved severe multicollinearity (which caused weight explosion) by introducing an optimized $L_2$ penalty (Diagonal Shifting).

## 📈 Model Progression & Performance Insights
The project follows a highly iterative, data-driven approach. The primary evaluation metric used is the **Root Mean Squared Error (RMSE)** measured in the log-space.

| Development Phase | Configuration | Validation RMSE |
| :--- | :--- | :--- |
| **1. Baseline Model** | 5 Numerical Features only | `0.73924` (High Bias) |
| **2. Feature Engineering** | Baseline + Engineered `age` | `0.50835` (Massive Drop) |
| **3. Full Pipeline (Singular)** | Full OHE (No Regularization) | `2.36 x 10^29` (Explosion) |
| **🏆 4. Champion Model** | **Full OHE + Ridge ($r = 0.1$)** | **`0.20594` (Optimal)** |

*The final evaluation on the fully isolated **Test Dataset** yielded an impressive **RMSE of 0.20491**, proving the model's robust generalization and lack of overfitting.*

## 🚀 Live Production Inference Demo
To simulate a real-world deployment, the champion model was tested on a single, unseen vehicle record extracted as a raw dictionary:
* **Vehicle Profile:** 2008 BMW Z4 M (Compact, Coupe, High-Performance)
* **Actual Market Price:** `$50,400`
* **Model's Prediction:** `$41,936.97`

> **Domain Insight:** The prediction falls perfectly within the statistical error bounds ($\pm15-20\%$). Considering the BMW Z4 M is a rare enthusiast car that carries a real-world scarcity premium (which pure linear depreciation models don't naturally capture), achieving a prediction this tight using a from-scratch analytical pipeline is a massive success.

## 🛠️ How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/AbdelrhmanHany3/Car-Price-Prediction-From-Scratch.git](https://github.com/AbdelrhmanHany3/Car-Price-Prediction-From-Scratch.git)
   ```
   Install the required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn category_encoders
   ```
   Run the Jupyter Notebook main.ipynb to execute the full pipeline from data exploratory analysis to the live inference demo.