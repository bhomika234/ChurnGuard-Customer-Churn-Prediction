import os
import joblib
import pandas as pd
import gradio as gr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "churn_model.joblib")

# Load the saved model artifact
artifact = joblib.load(MODEL_PATH)

# Support either a saved dictionary or a direct model
if isinstance(artifact, dict):
    model = artifact.get("model")
    FINAL_THRESHOLD = artifact.get("threshold", 0.5)
    FEATURES = artifact.get("feature_columns", [
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges"
    ])
else:
    model = artifact
    FINAL_THRESHOLD = 0.5
    FEATURES = [
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges"
    ]

if model is None:
    raise ValueError(
        "No model found in churn_model.joblib. "
        "Check train.py and retrain the model."
    )


def predict_customer(
    gender, senior, partner, dependents, tenure,
    phone, multiple, internet, security, backup,
    protection, support, tv, movies, contract,
    paperless, payment, monthly, total
):
    try:
        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": int(senior),
            "Partner": partner,
            "Dependents": dependents,
            "tenure": float(tenure),
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": float(monthly),
            "TotalCharges": float(total)
        }])

        customer = customer[FEATURES]
        probability = float(model.predict_proba(customer)[0, 1])

        prediction = (
            "Likely to Churn"
            if probability >= FINAL_THRESHOLD
            else "Likely to Stay"
        )

        return (
            f"Prediction: {prediction}\n"
            f"Churn probability: {probability:.1%}\n"
            f"Decision threshold: {FINAL_THRESHOLD:.2f}"
        )

    except Exception as e:
        return f"Prediction error: {e}"


def predict_csv(file):
    if file is None:
        return pd.DataFrame({
            "Error": ["Please upload a CSV file."]
        })

    try:
        df = pd.read_csv(file)

        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            return pd.DataFrame({
                "Error": [
                    "Missing required columns: " + ", ".join(missing)
                ]
            })

        X = df[FEATURES].copy()
        probabilities = model.predict_proba(X)[:, 1]

        results = df.copy()
        results["ChurnProbability"] = probabilities
        results["Prediction"] = [
            "Likely to Churn" if p >= FINAL_THRESHOLD
            else "Likely to Stay"
            for p in probabilities
        ]
        results["DecisionThreshold"] = FINAL_THRESHOLD

        return results

    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})


with gr.Blocks(title="ChurnGuard") as demo:
    gr.Markdown("# ChurnGuard")
    gr.Markdown(
        "Customer churn prediction and retention support tool."
    )
    gr.Markdown(
        f"**Decision threshold:** {FINAL_THRESHOLD:.2f}"
    )

    with gr.Tab("Single Customer"):
        inputs = [
            gr.Dropdown(["Female", "Male"], label="Gender", value="Female"),
            gr.Dropdown([0, 1], label="Senior Citizen", value=0),
            gr.Dropdown(["Yes", "No"], label="Partner", value="Yes"),
            gr.Dropdown(["Yes", "No"], label="Dependents", value="No"),
            gr.Slider(0, 72, value=12, step=1, label="Tenure (months)"),
            gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes"),
            gr.Dropdown(
                ["No phone service", "No", "Yes"],
                label="Multiple Lines", value="No"
            ),
            gr.Dropdown(
                ["DSL", "Fiber optic", "No"],
                label="Internet Service", value="Fiber optic"
            ),
            gr.Dropdown(
                ["No internet service", "No", "Yes"],
                label="Online Security", value="No"
            ),
            gr.Dropdown(
                ["No internet service", "No", "Yes"],
                label="Online Backup", value="No"
            ),
            gr.Dropdown(
                ["No internet service", "No", "Yes"],
                label="Device Protection", value="No"
            ),
            gr.Dropdown(
                ["No internet service", "No", "Yes"],
                label="Tech Support", value="No"
            ),
            gr.Dropdown(
                ["No internet service", "No", "Yes"],
                label="Streaming TV", value="No"
            ),
            gr.Dropdown(
                ["No internet service", "No", "Yes"],
                label="Streaming Movies", value="No"
            ),
            gr.Dropdown(
                ["Month-to-month", "One year", "Two year"],
                label="Contract", value="Month-to-month"
            ),
            gr.Dropdown(
                ["Yes", "No"], label="Paperless Billing", value="Yes"
            ),
            gr.Dropdown(
                [
                    "Electronic check", "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ],
                label="Payment Method", value="Electronic check"
            ),
            gr.Number(value=70.0, label="Monthly Charges"),
            gr.Number(value=840.0, label="Total Charges")
        ]

        predict_button = gr.Button("Predict Churn")
        output = gr.Textbox(label="Prediction Result", lines=4)

        predict_button.click(
            fn=predict_customer,
            inputs=inputs,
            outputs=output
        )

    with gr.Tab("CSV Upload"):
        gr.Markdown(
            "Upload a CSV containing all required customer features."
        )
        csv_input = gr.File(
            label="Upload customer CSV",
            file_types=[".csv"],
            type="filepath"
        )
        csv_button = gr.Button("Predict CSV Customers")
        csv_output = gr.Dataframe(label="Prediction Results")

        csv_button.click(
            fn=predict_csv,
            inputs=csv_input,
            outputs=csv_output
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
