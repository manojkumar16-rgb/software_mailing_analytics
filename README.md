# software_mailing_analytics
# 📊 Customer Purchase & Spending Prediction
## Predictive Analytics for Marketing Optimization

**Author:** Manoj Kumar  

---

## 📌 Project Overview

North-Point Software conducted a marketing mailing campaign to 20,000 customers and achieved only a **5.3% response rate**. Since mailing campaigns are costly, sending promotions to unlikely customers results in wasted resources and reduced profitability.

This project builds predictive models to:

- Predict which customers will make a purchase
- Predict which purchasers will spend more
- Identify high-value customer segments
- Improve marketing ROI through data-driven targeting

---

## 🎯 Business Problem

- Only **1,065 out of 20,000 customers** responded (5.3%)
- 94% of mailing effort produced no revenue
- Not all purchasers generate equal value
- No targeting system existed to identify high-value customers

---

## 🎯 Business Goals

✔ Increase campaign response rate  
✔ Improve average revenue per mailing  
✔ Reduce wasted marketing cost  
✔ Establish a repeatable data-driven targeting framework  

---

## 📊 Dataset Information

- **2,000 records (Stratified Sample)**
  - 1,000 Purchasers
  - 1,000 Non-Purchasers
- True population purchase rate: **5.3%**

### Key Variables

**Demographic Features**
- US
- Gender
- Address Type

**Behavioral Features**
- Freq (Transaction Frequency)
- last_update_days_ago
- 1st_update_days_ago
- Web Order History

**Source Indicators**
- source_a to source_w (Customer acquisition source)

**Target Variables**
- `Purchase` (Yes/No)
- `Spending`
- `SpendingClass` (Low / High)

---

## 🧹 Data Preprocessing

- Removed identifier column
- Converted binary variables into categorical factors
- Created derived variable: `SpendingClass`
- Verified:
  - No missing values
  - No duplicates
  - No negative or unrealistic entries
- Applied 70/30 stratified train-test split
- Used 5-fold cross-validation during training

---

## 📈 Modeling Approach

### 1️⃣ Purchase Prediction (Binary Classification)

Models Tested:
- Logistic Regression
- CART (Decision Tree)
- Random Forest
- Gradient Boosting Machine (GBM)

### 2️⃣ Spending Prediction (Low vs High)

Models Tested:
- Logistic Regression
- CART
- Random Forest
- Gradient Boosting Machine (GBM)

### 3️⃣ Customer Segmentation

- K-Means Clustering

---

## 🏆 Model Performance

### 📌 Purchase Prediction

| Model | AUC | Accuracy |
|-------|------|----------|
| Logistic Regression | 0.885 | 79% |
| CART | 0.863 | 78% |
| Random Forest | 0.896 | 82% |
| **GBM** | **0.908** | **82%** |

✔ GBM performed best  
✔ Balanced sensitivity & specificity  
✔ No overfitting observed  

---

### 📌 Spending Prediction

| Model | AUC | Accuracy |
|-------|------|----------|
| Logistic Regression | 0.730 | 67% |
| CART | 0.716 | 66% |
| Random Forest | 0.762 | 69% |
| **GBM** | **0.778** | **73%** |

✔ GBM again performed best  

---

## 🔍 Key Insights

### 1️⃣ Frequency (Freq) is the Strongest Predictor
Customers with higher past purchase frequency are:
- More likely to respond
- More likely to spend more

### 2️⃣ Recency Matters
Customers with more recent updates:
- Show higher engagement
- Have higher purchase probability

### 3️⃣ Source Channels Matter
Certain acquisition sources produce:
- Higher response rates
- Higher spending customers

### 4️⃣ Three Natural Customer Segments

| Cluster | Behavior | Purchase Rate |
|----------|----------|---------------|
| Cluster 1 | Low frequency, older updates | ~43% |
| Cluster 2 | Moderate activity | ~47% |
| Cluster 3 | High frequency, recent updates | ~93% |

Cluster 3 represents the ideal marketing target.

---

## 🚀 Business Impact

This predictive system enables:

- Ranking customers by purchase probability
- Targeting high-probability customers only
- Prioritizing high-spending purchasers
- Reducing mailing waste
- Increasing campaign profitability

---

## 🛠️ Tech Stack

- R Programming
- Logistic Regression
- Decision Trees (CART)
- Random Forest
- Gradient Boosting (GBM)
- K-Means Clustering
- ROC/AUC Evaluation
- Cross-validation

---

## 📌 Recommended Targeting Workflow

1. Score full customer list using GBM Purchase Model
2. Select top probability segments (e.g., top deciles)
3. Apply Spending Model within shortlisted group
4. Prioritize High SpendingClass customers
5. Use clustering for message personalization

---

## 📌 Conclusion

This project demonstrates how predictive analytics transforms traditional mailing campaigns into a **data-driven marketing optimization system**.

By combining:
- Gradient Boosting Models
- Customer Probability Scoring
- Spending Tier Classification
- Customer Segmentation

North-Point Software can significantly increase response rates, reduce costs, and maximize revenue from future campaigns.

---

⭐ If you found this project useful, feel free to star the repository.
