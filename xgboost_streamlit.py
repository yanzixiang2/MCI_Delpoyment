import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import datetime
selected_columns = [
    'X', 'Y', 'DIVISION', 'LOCATION_TYPE', 'PREMISES_TYPE', 'UCR_EXT',
    'HOOD_158', 'NEIGHBOURHOOD_158', 'HOOD_140', 'NEIGHBOURHOOD_140', 'OCC_YEAR', 'OCC_MONTH', 'OCC_DAY', 'OCC_DOY',
    'OCC_DOW',
    'OCC_HOUR'

]
# Load the saved model
best_model = joblib.load('C:/Users/Andrew/Downloads/xgb_model.joblib')
df = pd.read_csv("C:/Users/Andrew/Downloads/Major_Crime_Indicators_Open_Data.csv")
# Streamlit app
st.title("XGBoost Model Deployment")

# Collect user input
st.header("Enter the details:")
X_input = {}
label_encoder = LabelEncoder()
for column in selected_columns:
    if column in ['X', 'Y']:
        # Use number_input for numeric input fields
        X_input[column] = st.number_input(f"Enter {column}", step=0.0001)
    elif column == 'OCC_DAY':
        # Use date_input for day input
        X_input[column] = st.date_input(f"Select {column}", min_value=datetime.date(1900, 1, 1), max_value=datetime.date(2100, 12, 31))
    elif column == 'OCC_MONTH':
        # Use selectbox for month input
        X_input[column] = st.selectbox(f"Select {column}", range(1, 13))
    elif column == 'OCC_YEAR':
        # Use number_input for year input
        X_input[column] = st.number_input(f"Enter {column}", min_value=1900, max_value=2100)
    else:
        # Create a list of unique values for the dropdown
        unique_values = [''] + sorted(df[column].unique().tolist())
        # Use selectbox for other categorical fields
        X_input[column] = st.selectbox(f"Select {column}", unique_values)

# Calculate OCC_DOY based on OCC_DAY, OCC_MONTH, OCC_YEAR
if X_input['OCC_DAY'] and X_input['OCC_MONTH'] and X_input['OCC_YEAR']:
    input_date = datetime.date(X_input['OCC_YEAR'], X_input['OCC_MONTH'], X_input['OCC_DAY'])
    X_input['OCC_DOY'] = input_date.timetuple().tm_yday

# Make a prediction
if st.button("Predict"):
    # Create a DataFrame with the input data
    input_df = pd.DataFrame([X_input])

    # One-hot encode categorical variables
    input_encoded = pd.get_dummies(input_df, columns=['OCC_DOW', 'OCC_MONTH', 'DIVISION', 'LOCATION_TYPE', 'PREMISES_TYPE', 'HOOD_158',
                                        'NEIGHBOURHOOD_158', 'HOOD_140', 'NEIGHBOURHOOD_140'])

    # Make a prediction using the loaded model
    prediction = best_model.predict(input_encoded)

    # Inverse transform prediction to get original label
    prediction_original = label_encoder.inverse_transform(prediction)[0]

    st.success(f"Prediction: {prediction_original}")


