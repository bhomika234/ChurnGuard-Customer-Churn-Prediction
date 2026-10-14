# ChurnGuard — Customer Churn Prediction & Retention Tool

ChurnGuard is a machine learning project that predicts whether a telecom customer is likely to leave a service. It helps businesses identify customers who may need retention support.

The application supports single-customer predictions and batch predictions from CSV files.

## Project Objectives

- Analyze customer churn patterns and key drivers.
- Train and evaluate machine learning models.
- Handle class imbalance using class weights and, where appropriate, SMOTE.
- Evaluate models using metrics beyond accuracy.
- Support business decisions using retention cost assumptions.
- Provide a user-friendly Gradio interface.

## Dataset

This project uses the Telco Customer Churn dataset.

- Total records: 7,043
- Target column: `Churn`
- Features include customer tenure, contract type, internet services, payment method, and monthly charges.
- Original class distribution: 5,174 customers who stayed and 1,869 customers who churned.

Customer identifiers are excluded from model training.

## Machine Learning Workflow

1. Data cleaning and exploratory data analysis.
2. Train-validation-test separation.
3. Numerical preprocessing using median imputation and standard scaling.
4. Categorical preprocessing using most-frequent imputation and one-hot encoding.
5. Model training and comparison.
6. Validation-based threshold selection.
7. Final evaluation on the held-out test set.
8. Prediction through the Gradio application.

## Evaluation Metrics

Accuracy alone is insufficient because it can hide poor performance on the minority churn class.

- **Precision:** How many customers predicted to churn actually churned.
- **Recall:** How many actual churn customers the model identified.
- **F1-score:** Balance between precision and recall.
- **PR-AUC:** Measures precision-recall performance and is useful for imbalanced classification.
- **Brier score:** Evaluates the quality of predicted probabilities when reported.

### Final Model Comparison

Fill this table with actual results from the final, consistent test evaluation.

| Model | Precision | Recall | F1-score | PR-AUC |
|---|---:|---:|---:|---:|
| Dummy baseline | TODO | TODO | TODO | TODO |
| Logistic Regression | TODO | TODO | TODO | TODO |
| Random Forest | TODO | TODO | TODO | TODO |
| XGBoost | TODO | TODO | TODO | TODO |

The final model and threshold should be selected using training and validation data. The held-out test set must be used only for final evaluation.

## Business Cost Assumptions

The initial business scenario assumes:

- Cost of a lost customer: **$200**
- Cost of a retention offer: **$20**
- Assumed retention-offer success rate: **30%**

These are illustrative assumptions, not verified real-world costs.

The selected validation threshold is **0.55**, with a reported simplified expected validation cost of **$54,400**. This cost estimate must use the same documented cost formula throughout the notebook, application, and business report.

## Application Features

- Single-customer churn prediction.
- CSV upload for batch predictions.
- Predicted churn probabilities.
- Basic retention decision support.

**Explainability status:** SHAP-based global and per-customer explanations must be implemented and verified in the final application before claiming that Top 3 SHAP reasons are available for every prediction.

## Project Structure

```text
ChurnGuard/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── 01_eda.ipynb
├── 02_modelling.ipynb
└── churn_model.joblib
```

Keep large datasets and unnecessary temporary files out of the repository.

## Installation

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Ensure that `churn_model.joblib` is in the location expected by `app.py`.

## Deployment

- GitHub repository: TODO
- Hugging Face Space: TODO
- Demo video (2–3 minutes): TODO

Test both single-customer predictions and CSV uploads after deployment.

## Limitations

- Predictions are estimates, not guarantees that a customer will leave.
- Cost assumptions may not reflect a specific business.
- Model performance can change when customer behavior or service offerings change.
- SHAP explanations describe model behavior and do not establish causation.
- The model should be monitored and evaluated on new data before business use.

## Future Improvements

- Complete and verify SHAP Top 3 explanations in both application tabs.
- Add probability calibration and cost-sensitive threshold selection.
- Improve the user interface and input validation.
- Monitor performance and data drift after deployment.

## Author

Bhoomika Kumari

Machine Learning Project — Week 2
