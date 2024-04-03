import streamlit as st
import pandas as pd
import folium
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

# Load the trained model
classifier_model = joblib.load('ridge_classifier_model.joblib')

# Function for data preprocessing
def preprocess_data(data):
    # Drop any missing values
    data.dropna(inplace=True)

    # Perform feature scaling
    scaler = StandardScaler()
    numerical_cols = ['number_of_pentavalent_doses_received', 'number_of_pneumococcal_doses_received', 
                      'number_of_rotavirus_doses_received', 'number_of_measles_doses_received', 
                      'number_of_polio_doses_received','latitude', 'longitude']
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    return data

# Function to load and preprocess the validation dataset
def load_validation_data(file_path):
    df = pd.read_csv(file_path)
    df = preprocess_data(df)
    return df

def main():
    st.title('Vaccination Status Prediction and Visualization')

    # Upload validation dataset
    st.subheader('Upload Validation Dataset')
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)

        # Preprocess and make predictions
        df = preprocess_data(df)
        y_pred = classifier_model.predict(df)
        df['Predicted_Vaccination_Status'] = y_pred

        # Mapping predictions to status labels
        status_mapping = {0: 'Full_Defaulter', 1: 'Partial_Defaulter', 2: 'Non_Defaulter'}
        df['Predicted_Status'] = [status_mapping[pred] for pred in y_pred]

        # Create map
        st.subheader('Predicted Vaccination Status Visualization')
        mean_lat = df['latitude'].mean()
        mean_long = df['longitude'].mean()
        vaccination_map = folium.Map(location=[mean_lat, mean_long], zoom_start=6)

        # Add points to the map based on the predicted vaccination status
        for idx, row in df.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                color=get_color(row['Predicted_Status']),
                fill=True,
                fill_color=get_color(row['Predicted_Status']),
                fill_opacity=0.7,
                popup=row['Predicted_Status']  # Adjust if you want to display something else in the popup
            ).add_to(vaccination_map)

        # Display the map
        folium_static(vaccination_map)

if __name__ == '__main__':
    main()
