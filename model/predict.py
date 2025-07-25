import pickle
import pandas as pd

# Load saved model and scaler
with open("outputs/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("outputs/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load feature names
with open("outputs/feature_names.txt", "r") as f:
    expected_columns = [line.strip() for line in f.readlines()]

# Define predict function
def predict_price(input_dict):
    input_full = {col: 0 for col in expected_columns}
    input_full.update(input_dict)
    input_df = pd.DataFrame([input_full])[expected_columns]
    input_scaled = scaler.transform(input_df)
    return model.predict(input_scaled)[0]

# Example usage
if __name__ == "__main__":
    sample_input = {
        '1stFlrSF': 856,
        '2ndFlrSF': 854,
        'BedroomAbvGr': 3,
        'BsmtFinSF1': 706,
        'BsmtUnfSF': 150,
        'GarageArea': 548,
        'GrLivArea': 1710,
        'LotArea': 8450,
        'TotalBsmtSF': 856,
        'YearBuilt': 2003,
        'FullBath': 2,
        'TotRmsAbvGrd': 8
    }

    try:
        prediction = predict_price(sample_input)
        print(f"Predicted house price: ${prediction:,.2f}")
    except Exception as e:
        print("Prediction failed:", e)