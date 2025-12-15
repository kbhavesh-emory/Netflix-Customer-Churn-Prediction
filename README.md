# 📊 **Project Summary: Netflix Customer Churn Prediction**

### 🧠 **Objective:**

To predict whether a Netflix customer is likely to churn (cancel their subscription) using demographic, usage, and subscription-related data. This enables the business to proactively retain users by identifying at-risk customers.

---

### 📁 **Dataset Overview:**

* **Source**: Kaggle – [Netflix Customer Churn Dataset](https://www.kaggle.com/datasets/abdulwadood11220/netflix-customer-churn-dataset)
* **Total Records**: \~5000 customers
* **Features**:

  * `age`, `gender`, `subscription_type`, `watch_hours`, `last_login_days`, `region`, `device`, `monthly_fee`
  * Target: `churned` (Yes/No)

---

### 🔧 **Steps Performed:**

#### ✅ 1. **Data Cleaning**

* Removed duplicates
* Standardized and renamed columns for readability
* Dropped irrelevant identifiers like `customer_id`

#### ✅ 2. **Data Preprocessing**

* Encoded categorical features using `LabelEncoder`
* Converted target variable `churned` to binary (Yes → 1, No → 0)
* Scaled numerical features using `StandardScaler`

#### ✅ 3. **Exploratory Data Analysis (EDA)**

* Correlation heatmap to detect feature relationships
* Class balance check
* Feature distribution plots

#### ✅ 4. **Model Training & Evaluation**

Trained and compared multiple classifiers:

* **Logistic Regression**
* **Random Forest**
* **XGBoost**

Used metrics:

* **Accuracy**
* **Precision, Recall, F1-score**
* **Confusion Matrix**

#### ✅ 5. **Hyperparameter Tuning**

* Used `GridSearchCV` to tune Random Forest and Logistic Regression
* Improved performance via optimal regularization and tree depth

#### ✅ 6. **Feature Importance Analysis**

* Visualized most influential features using:

  * `.feature_importances_` (Tree models)
  * `.coef_` (Logistic Regression)

#### ✅ 7. **Model Deployment**

* Deployed the best model (XGBoost) using **AWS SageMaker** as a REST API
* Exposed predictions via a scalable, cloud-hosted endpoint
* Stored pre-trained model and scaler using `joblib`

---

### 🏆 **Best Performing Model: XGBoost**

| Metric    | Score     |
| --------- | --------- |
| Accuracy  | **99.5%** |
| Precision | 1.00      |
| Recall    | 0.99      |
| F1-score  | 1.00      |

---

### 💡 **Key Business Insights:**

* **Churn highly correlates with low watch hours and infrequent login activity**
* **Subscription type and device usage patterns** also influence churn
* XGBoost provides **high precision**, minimizing false churn predictions, making it ideal for retention strategies

---

### 🛠️ **Technologies Used:**

* **Python**, **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**
* **Scikit-learn**, **XGBoost**
* **AWS SageMaker**, **S3**, **joblib**

---

### 📦 **Deliverables:**

* Cleaned & labeled dataset
* Trained & tuned models
* Model evaluation report
* Deployment-ready API on SageMaker
* Feature importance visualizations

---

### 📁 \[Optional GitHub Structure]

```
netflix-churn-prediction/
├── data/
├── notebooks/
├── models/
│   └── xgboost_model.pkl
├── scaler.pkl
├── inference.py
├── sagemaker_deploy.py
└── README.md
```
### Thank you ! 😊👍
