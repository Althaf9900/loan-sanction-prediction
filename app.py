import numpy as np
import pickle as pkl
import streamlit as st

# Load scaler
def load_scaler(file_path):
    with open(file_path, 'rb') as f:
        return pkl.load(f)

# Load encoder
def load_encoder(file_path):
    with open(file_path, 'rb') as f:
        return pkl.load(f)

# Load model
def load_model(file_path):
    with open(file_path, 'rb') as f:
        return pkl.load(f)

# Function to make predictions
def predict_loan_approval(classifier, final_input):
    return classifier.predict(final_input)

# Main function for the Streamlit app
def main():
    # Load the scaler, encoder, and model
    scaler = load_scaler('scaler.pkl')
    encoder = load_encoder('encoder.pkl')
    classifier = load_model('model.pkl')

    # Set the title of the web page
    st.title("Loan Sanction Prediction")

    # Define min and max values for numerical inputs
    min_value = float(np.finfo(float).tiny)
    max_value = float(np.finfo(float).max)

    # Collect user inputs
    gender = st.radio("Gender", ("Male", "Female"), horizontal=True)
    
    married = st.radio("Marital Status", ("Married", "Unmarried"), horizontal=True)
    married = 'Yes' if married == 'Married' else 'No'
    
    dependents = st.radio("Number of Dependents", ("0", "1", "2", "3+"), horizontal=True)
    education = st.radio("Education", ("Graduate", "Not Graduate"))
    self_employed = st.radio("Are you self-employed?", ("Yes", "No"))
    applicant_income = st.number_input("Applicant's Income", min_value=min_value, max_value=max_value)

    has_coapplicant = st.radio("Do you have a co-applicant?", ("Yes", "No"))

    # Initialize co-applicant income
    coapplicant_income = 0.0
    if has_coapplicant == 'Yes':
        coapplicant_income = st.number_input("Co-applicant's Income", min_value=min_value, max_value=max_value)

    loan_amount = st.number_input("Loan Amount", min_value=min_value, max_value=max_value)

    # Select loan amount term from predefined options
    loan_amount_terms = [12, 36, 60, 84, 120, 180, 240, 300, 360, 480]
    loan_amount_term = st.selectbox("Select Loan Amount Term", loan_amount_terms)

    credit_history = st.radio("Credit History", (0, 1))
    property_area = st.radio("Property Area", ("Rural", "Semi-urban", "Urban"))

    if property_area == 'Semi-urban':
        property_area = 'Semiurban'

    # When the 'Predict' button is clicked
    if st.button("Predict"):
        # Prepare numerical features for scaling
        num_cols = np.array([[applicant_income, coapplicant_income, loan_amount]])

        # Prepare categorical features for encoding
        cat_cols = np.array([[gender, married, dependents, education, self_employed, 
                              loan_amount_term, credit_history, property_area, has_coapplicant]])

        # Scale numerical features
        scaled_features = scaler.transform(num_cols)

        # Encode categorical features
        encoded_features = encoder.transform(cat_cols)

        # Combine scaled numerical and encoded categorical features
        final_input = np.hstack((scaled_features, encoded_features))

        # Get prediction
        result = predict_loan_approval(classifier, final_input)

        # Display the result
        if result == 1:
            st.success("The loan can be sanctioned.")
        elif result == 0:
            st.error("The loan cannot be sanctioned.")

if __name__ == '__main__':
    main()