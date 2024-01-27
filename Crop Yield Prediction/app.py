import streamlit as st
import numpy as np
import pickle

# Load the pre-trained models
preprocesser = pickle.load(open('preprocessor.pkl', 'rb'))
model = pickle.load(open('KNN.pkl', 'rb'))

# Function to preprocess input data
def preprocess_input(data):
    # Your preprocessing steps here
    # Ensure that the preprocessing steps match the ones used during training
    # For example:
    # data = preprocesser.transform(data)
    return data

# Function to make predictions
def predict_yield(features):
    features = np.array(features).reshape(1, -1)
    # Ensure that the input data matches the expected format and has the correct number of features
    # For example, if your model expects 75 features, ensure that `features` has 75 elements
    features = preprocess_input(features)
    prediction = model.predict(features)
    return prediction

# Streamlit App
def main():
    st.title('Crop Yield Prediction')

    # Input features
    crop_year = st.selectbox('Crop Year', list(range(1990, 2024)))
    avg_rainfall = st.number_input('Average Rainfall (mm per year)', value=1000)
    pesticides = st.number_input('Pesticides (tonnes)', value=500)
    avg_temp = st.number_input('Average Temperature', value=20.0)
    area = st.selectbox('Area', ['Albania', 'Algeria', 'Angola', 'Argentina', 'Armenia',
                                 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain',
                                 'Bangladesh', 'Belarus', 'Belgium', 'Botswana', 'Brazil',
                                 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cameroon', 'Canada',
                                 'Central African Republic', 'Chile', 'Colombia', 'Croatia',
                                 'Denmark', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador',
                                 'Eritrea', 'Estonia', 'Finland', 'France', 'Germany', 'Ghana',
                                 'Greece', 'Guatemala', 'Guinea', 'Guyana', 'Haiti', 'Honduras',
                                 'Hungary', 'India', 'Indonesia', 'Iraq', 'Ireland', 'Italy',
                                 'Jamaica', 'Japan', 'Kazakhstan', 'Kenya', 'Latvia', 'Lebanon',
                                 'Lesotho', 'Libya', 'Lithuania', 'Madagascar', 'Malawi',
                                 'Malaysia', 'Mali', 'Mauritania', 'Mauritius', 'Mexico',
                                 'Montenegro', 'Morocco', 'Mozambique', 'Namibia', 'Nepal',
                                 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Norway',
                                 'Pakistan', 'Papua New Guinea', 'Peru', 'Poland', 'Portugal',
                                 'Qatar', 'Romania', 'Rwanda', 'Saudi Arabia', 'Senegal',
                                 'Slovenia', 'South Africa', 'Spain', 'Sri Lanka', 'Sudan',
                                 'Suriname', 'Sweden', 'Switzerland', 'Tajikistan', 'Thailand',
                                 'Tunisia', 'Turkey', 'Uganda', 'Ukraine', 'United Kingdom',
                                 'Uruguay', 'Zambia', 'Zimbabwe'])
    item = st.selectbox('Item', ['Maize', 'Potatoes', 'Rice, paddy', 'Sorghum', 'Soybeans', 'Wheat',
                                  'Cassava', 'Sweet potatoes', 'Plantains and others', 'Yams'])

    # Prediction button
    if st.button('Predict Yield'):
        input_data = [crop_year, avg_rainfall, pesticides, avg_temp, area, item]

        # Ensure that all input features are provided
        if '' in input_data:
            st.error('Please fill in all the input fields.')
        else:
            # Ensure that the input data matches the expected format and has the correct number of features
            # For example, if your model expects 75 features, ensure that `input_data` has 75 elements
            prediction = predict_yield(input_data)
            st.success(f'Predicted yield (hg/ha): {prediction[0]:.2f}')

if __name__ == '__main__':
    main()
