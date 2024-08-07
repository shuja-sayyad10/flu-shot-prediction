import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px



# Load the model
model = joblib.load('C:/Users/acer/OneDrive/Desktop/dic/manaliba_sshahlal_tajammul_phase_3/src/multioutput_model3.pkl')

# Title and description
st.title('Flu Shot Learning')
st.write('Predicts the likelihood of an individual having taken H1N1 and seasonal flu vaccines.')

def transform_df(df):
    numerical_columns = ['h1n1_concern', 'h1n1_knowledge', 'doctor_recc_h1n1', 'doctor_recc_seasonal',
                         'chronic_med_condition', 'health_worker',
                         'health_insurance', 'opinion_h1n1_vacc_effective', 'opinion_h1n1_risk',
                         'opinion_h1n1_sick_from_vacc', 'opinion_seas_vacc_effective',
                         'opinion_seas_risk', 'opinion_seas_sick_from_vacc']


    # Fill missing values
    for column in numerical_columns:
        if df[column].isnull().any():
            mode_val = df[column].mode()[0]
            df[column].fillna(mode_val, inplace=True)

    # One-hot encode the categorical columns only
    #df = pd.get_dummies(df, columns=categ_columns, drop_first=True)

    return df


h1n1_concern_options = [
    'Not at all concerned',
    'Not very concerned',
    'Somewhat concerned',
    'Very concerned'
]
h1n1_concern = st.selectbox('h1n1_concern', h1n1_concern_options)
h1n1_concern_mapping = {
    'Not at all concerned': 0,
    'Not very concerned': 1,
    'Somewhat concerned': 2,
    'Very concerned': 3
}
h1n1_concern_value = h1n1_concern_mapping[h1n1_concern]

h1n1_knowledge_options = [
    'No knowledge',
    'A little knowledge',
    'A lot of knowledge'
]
h1n1_knowledge = st.selectbox('h1n1_knowledge', h1n1_knowledge_options)
h1n1_knowledge_mapping = {
    'No knowledge': 0,
    'A little knowledge': 1,
    'A lot of knowledge': 2
}
h1n1_knowledge_value = h1n1_knowledge_mapping[h1n1_knowledge]

doctor_recc_h1n1_options = [
    'Not Recommended',
    'Recommended'
]
doctor_recc_h1n1 = st.selectbox('Doctor Recommendation for H1N1 Vaccine', doctor_recc_h1n1_options)
doctor_recc_h1n1_mapping = {
    'Not Recommended': 0,
    'Recommended': 1
}
doctor_recc_h1n1_value = doctor_recc_h1n1_mapping[doctor_recc_h1n1]


doctor_recc_seasonal_options = [
    'Not Recommended',
    'Recommended'
]
doctor_recc_seasonal = st.selectbox('Doctor Recommendation for Seasonal Flu Vaccine', doctor_recc_seasonal_options)
doctor_recc_seasonal_mapping = {
    'Not Recommended': 0,
    'Recommended': 1
}
doctor_recc_seasonal_value = doctor_recc_seasonal_mapping[doctor_recc_seasonal]


chronic_med_condition_options = [
    'No',
    'Yes'
]
chronic_med_condition = st.selectbox('Chronic Medical Condition', chronic_med_condition_options)
chronic_med_condition_mapping = {
    'No': 0,
    'Yes': 1
}
chronic_med_condition_value = chronic_med_condition_mapping[chronic_med_condition]


health_worker_options = [
    'No',
    'Yes'
]
health_worker = st.selectbox('Healthcare Worker', health_worker_options)
health_worker_mapping = {
    'No': 0,
    'Yes': 1
}
health_worker_value = health_worker_mapping[health_worker]


health_insurance_options = [
    'No',
    'Yes'
]
health_insurance = st.selectbox('Has Health Insurance', health_insurance_options)
health_insurance_mapping = {
    'No': 0,
    'Yes': 1
}
health_insurance_value = health_insurance_mapping[health_insurance]


opinion_h1n1_vacc_effective_options = [
    'Not at all effective',
    'Not very effective',
    "Don't know",
    'Somewhat effective',
    'Very effective'
]
opinion_h1n1_vacc_effective = st.selectbox('Opinion on H1N1 Vaccine Effectiveness', opinion_h1n1_vacc_effective_options)
opinion_h1n1_vacc_effective_mapping = {
    'Not at all effective': 1,
    'Not very effective': 2,
    "Don't know": 3,
    'Somewhat effective': 4,
    'Very effective': 5
}
opinion_h1n1_vacc_effective_value = opinion_h1n1_vacc_effective_mapping[opinion_h1n1_vacc_effective]


opinion_h1n1_risk_options = [
    'Very Low',
    'Somewhat low',
    "Don't know",
    'Somewhat high',
    'Very high'
]
opinion_h1n1_risk = st.selectbox('Opinion on Risk of H1N1 Without Vaccine', opinion_h1n1_risk_options)
opinion_h1n1_risk_mapping = {
    'Very Low': 1,
    'Somewhat low': 2,
    "Don't know": 3,
    'Somewhat high': 4,
    'Very high': 5
}
opinion_h1n1_risk_value = opinion_h1n1_risk_mapping[opinion_h1n1_risk]


opinion_h1n1_sick_from_vacc_options = [
    'Not at all worried',
    'Not very worried',
    "Don't know",
    'Somewhat worried',
    'Very worried'
]
opinion_h1n1_sick_from_vacc = st.selectbox('Worry About Getting Sick from H1N1 Vaccine', opinion_h1n1_sick_from_vacc_options)
opinion_h1n1_sick_from_vacc_mapping = {
    'Not at all worried': 1,
    'Not very worried': 2,
    "Don't know": 3,
    'Somewhat worried': 4,
    'Very worried': 5
}
opinion_h1n1_sick_from_vacc_value = opinion_h1n1_sick_from_vacc_mapping[opinion_h1n1_sick_from_vacc]


opinion_seas_vacc_effective_options = [
    'Not at all effective',
    'Not very effective',
    "Don't know",
    'Somewhat effective',
    'Very effective'
]
opinion_seas_vacc_effective = st.selectbox('Opinion on Seasonal Flu Vaccine Effectiveness', opinion_seas_vacc_effective_options)
opinion_seas_vacc_effective_mapping = {
    'Not at all effective': 1,
    'Not very effective': 2,
    "Don't know": 3,
    'Somewhat effective': 4,
    'Very effective': 5
}
opinion_seas_vacc_effective_value = opinion_seas_vacc_effective_mapping[opinion_seas_vacc_effective]


opinion_seas_risk_options = [
    'Very Low',
    'Somewhat low',
    "Don't know",
    'Somewhat high',
    'Very high'
]
opinion_seas_risk = st.selectbox('Opinion on Risk of Seasonal Flu Without Vaccine', opinion_seas_risk_options)
opinion_seas_risk_mapping = {
    'Very Low': 1,
    'Somewhat low': 2,
    "Don't know": 3,
    'Somewhat high': 4,
    'Very high': 5
}
opinion_seas_risk_value = opinion_seas_risk_mapping[opinion_seas_risk]


opinion_seas_sick_from_vacc_options = [
    'Not at all worried',
    'Not very worried',
    "Don't know",
    'Somewhat worried',
    'Very worried'
]
opinion_seas_sick_from_vacc = st.selectbox('Worry About Getting Sick from Seasonal Flu Vaccine', opinion_seas_sick_from_vacc_options)
opinion_seas_sick_from_vacc_mapping = {
    'Not at all worried': 1,
    'Not very worried': 2,
    "Don't know": 3,
    'Somewhat worried': 4,
    'Very worried': 5
}
opinion_seas_sick_from_vacc_value = opinion_seas_sick_from_vacc_mapping[opinion_seas_sick_from_vacc]

# Button for predictions

# # Create a DataFrame from the collected inputs
if st.button('Predict'):
    input_data = pd.DataFrame([{
        'h1n1_concern': h1n1_concern_value,
        'h1n1_knowledge': h1n1_knowledge_value,
        'doctor_recc_h1n1': doctor_recc_h1n1_value,
        'doctor_recc_seasonal': doctor_recc_seasonal_value,
        'chronic_med_condition': chronic_med_condition_value,
        'health_worker': health_worker_value,
        'health_insurance': health_insurance_value,
        'opinion_h1n1_vacc_effective': opinion_h1n1_vacc_effective_value,
        'opinion_h1n1_risk': opinion_h1n1_risk_value,
        'opinion_h1n1_sick_from_vacc': opinion_h1n1_sick_from_vacc_value,
        'opinion_seas_vacc_effective': opinion_seas_vacc_effective_value,
        'opinion_seas_risk': opinion_seas_risk_value,
        'opinion_seas_sick_from_vacc': opinion_seas_sick_from_vacc_value

    }], columns=[
        'h1n1_concern', 'h1n1_knowledge', 'doctor_recc_h1n1', 'doctor_recc_seasonal',
        'chronic_med_condition', 'health_worker', 'health_insurance',
        'opinion_h1n1_vacc_effective', 'opinion_h1n1_risk', 'opinion_h1n1_sick_from_vacc',
        'opinion_seas_vacc_effective', 'opinion_seas_risk', 'opinion_seas_sick_from_vacc'
    ])

    # Apply the transformation function if any additional preprocessing is required
    transformed_input = transform_df(input_data)

    # Make prediction
    prediction = model.predict(transformed_input)

    # Extract prediction results
    h1n1_vaccine_uptake = prediction[0][0]
    seasonal_vaccine_uptake = prediction[0][1]

    # Display prediction
    st.write(f'Predicted H1N1 Vaccine Uptake: {prediction[0][0]}')
    st.write(f'Predicted Seasonal Flu Vaccine Uptake: {prediction[0][1]}')

    # Conditional messages based on prediction
    if h1n1_vaccine_uptake == 0:
        st.warning('Based on your responses, it seems you may not have received the H1N1 vaccine. Please consider getting vaccinated for your safety and public health.')

    if seasonal_vaccine_uptake == 0:
        st.warning('Our prediction suggests you may not have received the seasonal flu vaccine. We encourage you to get vaccinated.')

    if h1n1_vaccine_uptake == 1:
        st.success('It appears that you have received the H1N1 vaccine. Thank you for taking this important step in protecting not only your health but also contributing to the wider community’s safety during the H1N1 pandemic.')

    if seasonal_vaccine_uptake == 1:
        st.success('Our prediction suggests that you have received the seasonal flu vaccine. Your action helps in safeguarding public health and controlling the spread of influenza. Thank you for your responsible decision.')


#
# df_features=pd.read_csv('training_set_features.csv')
# df_labels=pd.read_csv('training_set_labels.csv')
# df = df_features.merge(df_labels, on='respondent_id')


# # Header for the statistics section
# st.header("Check the Statistics of People")

# # Button to show statistics
# if st.button("Show Statistics"):
#     # Count the number of people vaccinated and not vaccinated
#     vaccine_counts = df[['h1n1_vaccine', 'seasonal_vaccine']].apply(pd.Series.value_counts).fillna(0)
#     vaccine_counts.index = ['Vaccinated', 'Not Vaccinated']

#     # Plotting using Plotly
#     fig = px.bar(vaccine_counts.T, x=vaccine_counts.T.index, y=['Vaccinated', 'Not Vaccinated'], 
#                   title='Vaccine Uptake', labels={'value':'Number of People', 'variable':'Vaccination Status'})
#     st.plotly_chart(fig)

#     # Display the raw statistics
#     st.write('Statistics after prediction:')
#     st.write(vaccine_counts)