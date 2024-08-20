import joblib
import numpy as np
import streamlit as st

# Main function for the Streamlit app
def main():
    # Load the preprocessor and classifier model
    preprocessor = joblib.load('preprocessor.joblib')
    classifier = joblib.load('model.joblib')

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
        data = np.array([[applicant_income, coapplicant_income, loan_amount, gender, 
        married, dependents, education, self_employed, loan_amount_term, credit_history, 
        property_area, has_coapplicant]])

        # Preprocess the data
        preprocessed_data = preprocessor.transform(data)

        # Predict the result
        result = classifier.predict(preprocessed_data)

        # Display the result
        if result == 1:
            st.success("The loan can be sanctioned.")
        elif result == 0:
            st.error("The loan cannot be sanctioned.")

if __name__ == '__main__':
    main()