# Import necessary packages
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the pre-trained model
@st.cache_resource
def load_model():
    return joblib.load('patient_model.pkl')

# Load data
@st.cache_resource
def load_data():
    df_train = pd.read_csv('Book2.csv')
    df_train.rename(columns=lambda x: x.strip(), inplace=True)  # Remove accidental spaces
    return df_train

# Preprocess input data
def preprocess_input(input_data, label_encoder):
    input_data['disease'] = label_encoder.transform(input_data['disease'])
    return input_data[['Age', 'disease']]

# Mortality rates dictionary
mortality_rates = {
    'influenza (flu)': 0.75,
    'diabetes': 0.34,
    'cancer (various types)': 0.9,
    'hypertension (high blood pressure)': 0.6,
    "alzheimer's disease": 0.47,
    "parkinson's disease": 0.65,
    'asthma': 0.5,
    'arthritis': 0.55,
    'chronic obstructive pulmonary disease (copd)': 0.6,
    'tuberculosis (tb)': 0.7,
    'hepatitis (various types)': 0.75,
    'malaria': 0.8,
    'hiv/aids': 0.85,
    'ebola virus disease': 0.39,
    'dengue fever': 0.7,
    'cholera': 0.6,
    'typhoid fever': 0.65,
    'lyme disease': 0.55,
    'multiple sclerosis (ms)': 0.8,
    'lupus (systemic lupus erythematosus)': 0.7,
    'rheumatoid arthritis': 0.25,
    'osteoporosis': 0.1,
    'celiac disease': 0.55,
    "crohn's disease": 0.65,
    'ulcerative colitis': 0.4,
    'endometriosis': 0.8,
    'fibromyalgia': 0.75,
    'chronic fatigue syndrome (cfs)': 0.7,
    'psoriasis': 0.6,
    'eczema (atopic dermatitis)': 0.55,
    'schizophrenia': 0.75,
    'bipolar disorder': 0.8,
    'obsessive-compulsive disorder (ocd)': 0.7,
    'post-traumatic stress disorder (ptsd)': 0.65,
    'attention deficit hyperactivity disorder (adhd)': 0.6,
    'autism spectrum disorder (asd)': 0.55,
    'down syndrome': 0.7,
    'turner syndrome': 0.15,
}

# Display message based on mortality rate
def display_message(mortality_rate):
    if mortality_rate >= 0.5:
        return "⚠️ **Condition: Severe. Requires further medical attention. Operation Required.**"
    else:
        return "✅ **Condition: Good. No significant medical intervention required. Operation not required.**"

# Main function
def main():
    page = st.sidebar.selectbox("Choose a page", ["Home", "Prediction", "About"])
    
    # Home Page
    if page == "Home":
        st.title("🏠 Home Page")
        st.image("ED-ICU.png")
        st.write("Welcome to the **Medical Condition Prediction App**!")
        
    # Prediction Page
    elif page == "Prediction":
        st.title('🩺 Condition Prediction')
        st.write("Enter patient information to predict the condition:")
        
        # Load data and model
        df_train = load_data()
        model = load_model()
        
        # Fit LabelEncoder
        label_encoder = LabelEncoder()
        df_train['disease'] = df_train['disease'].astype(str)  # Ensure string type
        label_encoder.fit(df_train['disease'])
        
        # User inputs
        age = st.number_input('Age', min_value=0, max_value=120, value=30)
        disease = st.selectbox('Disease', df_train['disease'].unique())
        
        # Predict button
        if st.button('🔍 Predict'):
            # Prepare input data
            input_data = {'Age': age, 'disease': disease}
            input_df = pd.DataFrame([input_data])
            
            # Preprocess input
            preprocessed_input = preprocess_input(input_df, label_encoder)
            
            # Make prediction
            predicted_condition = model.predict(preprocessed_input)[0]
            
            # Display results
            st.subheader("📊 Prediction Results")
            st.write(f'**Age**: {age}')
            st.write(f'**Disease**: {disease}')
            
            # Lookup mortality rate
            disease_lower = disease.lower()
            if disease_lower in mortality_rates:
                mortality_rate = mortality_rates[disease_lower]
                st.write(f'🧬 **Mortality Rate for {disease}**: {mortality_rate * 100:.2f}%')
                
                # Display condition message
                message = display_message(mortality_rate)
                st.markdown(message)
            else:
                st.warning("Mortality rate data not available for this disease.")

    # About Page
    elif page == "About":
        st.title("ℹ️ About")
        st.write("This is a **Streamlit** app for predicting medical conditions based on patient information.")
        st.write("It uses a **pre-trained Random Forest** classifier to make predictions.")
        st.write("The app allows users to **input patient age and disease** and then predicts the medical condition.")
        st.write("It also provides **mortality rate information** for selected diseases.")
        st.write("⚕️ **This app is intended to assist healthcare professionals in making informed decisions.**")

if __name__ == '__main__':
    main()
