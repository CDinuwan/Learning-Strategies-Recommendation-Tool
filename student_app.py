import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Learning Strategy Recommendation System

This app predicts the **Student Performance** in the University!
And recommend suitable learning strategies for the students.
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input File]
""")

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        marital_status = st.sidebar.selectbox('Marital status', (1, 2, 3, 4, 5, 6))
        gender = st.sidebar.selectbox('Gender', (1, 0))
        application_order = st.sidebar.slider('Application order', 0, 9, 3)
        age_at_enrollment = st.sidebar.slider('Age at enrollment', 0, 100, 25)
        daytime_evening_attendance = st.sidebar.selectbox('Daytime/evening_attendance', (1, 0))
        displaced = st.sidebar.selectbox('Displaced', (1, 0))
        educational_special_needs = st.sidebar.selectbox('Educational special needs', (1, 0))
        debtor = st.sidebar.selectbox('Debtor', (1, 0))
        tuition_fees_up_to_date = st.sidebar.selectbox('Tuition fees up to date', (1, 0))
        scholarship_holder = st.sidebar.selectbox('Scholarship holder', (1, 0))
        international = st.sidebar.selectbox('International', (1, 0))
        curricular_units_1st_sem_credited = st.sidebar.slider('Curricular units 1st sem (credited)', 0, 50, 15)
        curricular_units_1st_sem_enrolled = st.sidebar.slider('Curricular units 1st sem (enrolled)', 0, 50, 15)
        curricular_units_1st_sem_evaluations = st.sidebar.slider('Curricular units 1st sem (evaluations)', 0, 50, 15)
        curricular_units_1st_sem_approved = st.sidebar.slider('Curricular units 1st sem (approved)', 0, 50, 15)
        curricular_units_1st_sem_grade = st.sidebar.slider('Curricular units 1st sem (grade)', 0, 50, 15)
        curricular_units_1st_sem_without_evaluations = st.sidebar.slider('Curricular units 1st sem (without evaluations)', 0, 50, 15)
        curricular_units_2nd_sem_credited = st.sidebar.slider('Curricular units 2nd sem (credited)', 0, 50, 15)
        curricular_units_2nd_sem_enrolled = st.sidebar.slider('Curricular units 2nd sem (enrolled)', 0, 50, 15)
        curricular_units_2nd_sem_evaluations = st.sidebar.slider('Curricular units 2nd sem (evaluations)', 0, 50, 15)
        curricular_units_2nd_sem_approved = st.sidebar.slider('Curricular units 2nd sem (approved)', 0, 50, 15)
        curricular_units_2nd_sem_grade = st.sidebar.slider('Curricular units 2nd sem (grade)', 0, 50, 15)
        curricular_units_2nd_sem_without_evaluations = st.sidebar.slider('Curricular units 2nd sem (without evaluations)', 0, 50, 15)
        unemployment_rate = st.sidebar.slider('Unemployment rate', 0.0, 50.0, 10.0)
        inflation_rate = st.sidebar.slider('Inflation rate', -10.0, 10.0, 0.0)
        gdp = st.sidebar.slider('GDP', -10.0, 10.0, 0.0)
        data = {'Marital status': marital_status,
                'Gender': gender,
                'Application order': application_order,
                'Age at enrollment': age_at_enrollment,
                'Daytime/evening_attendance': daytime_evening_attendance,
                'Displaced': displaced,
                'Educational special needs': educational_special_needs,
                'Debtor': debtor,
                'Tuition fees up to date': tuition_fees_up_to_date,
                'Scholarship holder': scholarship_holder,
                'International': international,
                'Curricular units 1st sem (credited)': curricular_units_1st_sem_credited,
                'Curricular units 1st sem (enrolled)': curricular_units_1st_sem_enrolled,
                'Curricular units 1st sem (evaluations)': curricular_units_1st_sem_evaluations,
                'Curricular units 1st sem (approved)': curricular_units_1st_sem_approved,
                'Curricular units 1st sem (grade)': curricular_units_1st_sem_grade,
                'Curricular units 1st sem (without evaluations)': curricular_units_1st_sem_without_evaluations,
                'Curricular units 2nd sem (credited)': curricular_units_2nd_sem_credited,
                'Curricular units 2nd sem (enrolled)': curricular_units_2nd_sem_enrolled,
                'Curricular units 2nd sem (evaluations)': curricular_units_2nd_sem_evaluations,
                'Curricular units 2nd sem (approved)': curricular_units_2nd_sem_approved,
                'Curricular units 2nd sem (grade)': curricular_units_2nd_sem_grade,
                'Curricular units 2nd sem (without evaluations)': curricular_units_2nd_sem_without_evaluations,
                'Unemployment rate': unemployment_rate,
                'Inflation rate': inflation_rate,
                'GDP': gdp,
                }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

student_raw = pd.read_csv(r'dataset.csv', sep=";")
student = student_raw.drop(columns=['Target'])
student = pd.concat([input_df, student], axis=0)

student['Curricular enrolled_u per sem'] = (student['Curricular units 1st sem (enrolled)'] + student['Curricular units 2nd sem (enrolled)'])/2
student['Curricular evaluation_u per sem'] = (student['Curricular units 1st sem (evaluations)'] + student['Curricular units 2nd sem (evaluations)'])/2
student['Curricular approved_u per sem'] = (student['Curricular units 1st sem (approved)'] + student['Curricular units 2nd sem (approved)'])/2
student['Curricular grade_u per sem'] = (student['Curricular units 1st sem (grade)'] + student['Curricular units 2nd sem (grade)'])/2
student['Curricular w/o_evaluation_u per sem'] = (student['Curricular units 1st sem (without evaluations)'] + student['Curricular units 2nd sem (without evaluations)'])/2

drop_colms = ['Curricular units 1st sem (credited)','Curricular units 1st sem (enrolled)','Curricular units 1st sem (evaluations)',
             'Curricular units 1st sem (approved)',"Curricular units 1st sem (grade)","Curricular units 1st sem (without evaluations)",
             'Curricular units 2nd sem (credited)','Curricular units 2nd sem (enrolled)','Curricular units 2nd sem (evaluations)',
             'Curricular units 2nd sem (approved)','Curricular units 2nd sem (grade)','Curricular units 2nd sem (without evaluations)'
            ]
student.drop(drop_colms, axis = 1, inplace = True)

encode = ['Marital status', 'Daytime/evening attendance', 'Displaced', 'Educational special needs',
          'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International']
for col in encode:
    dummy = pd.get_dummies(student[col], prefix=col)
    student = pd.concat([student, dummy], axis=1)
    del student[col]

drop_dcols = ['Marital status_6','Daytime/evening attendance_0.0','Daytime/evening attendance_1.0','Displaced_0','Educational special needs_0','Debtor_0',
              'Tuition fees up to date_0','Gender_0','Scholarship holder_0','International_0','Application mode','Course','Previous qualification','Nacionality',"Mother's qualification",
             "Father's qualification", "Mother's occupation","Father's occupation"]
student.drop(drop_dcols, axis = 1, inplace = True)

student = student[:1]

st.subheader('User Input features')

if uploaded_file is not None:
    st.write(student)
else:
    st.write('Awaiting CSV file to be uploaded')
    st.write(student)
    st.write('---')

load_clf = pickle.load(open('student_performance_predictor_model.pkl', 'rb'))

prediction = load_clf.predict(student)
prediction_proba = load_clf.predict_proba(student)


st.subheader('Prediction')
student_result = np.array(['Dropout', 'Enrolled', 'Graduate'])
col = ['Result']
results = pd.DataFrame(student_result[prediction], columns = col)
st.write(results)
st.write('---')


if student_result[prediction] == 'Dropout':
    st.subheader('Student is likely to dropout')
    st.write('---')
    st.write('Learning Strategies for Dropout Students')
    st.write('1. Enroll them in a learning skills course that teaches effective study strategies like self-testing, note-taking, time management, and self-regulation.')
    st.write('2. Provide intensive academic advising and follow-up to ensure they apply the learned strategies consistently.')
    st.write('3. Encourage active engagement strategies like creating note cards, rewriting notes, and self-quizzing instead of passive rereading.')
    st.write('4. Help them develop better time management skills and dedicate enough study hours per week.')
    st.write('5. Address any motivational issues or negative attitudes towards challenging content that may lead them to disengage.')
    st.write('---')
elif student_result[prediction] == 'Enrolled':
    st.subheader('Student is likely to stay enrolled but not completing')
    st.write('---')
    st.write('Learning Strategies for Enrolled Students')
    st.write('1. Assess their current study skills and identify gaps or ineffective strategies like excessive rereading.')
    st.write('2. Introduce a variety of active learning strategies tailored to their needs and learning styles.')
    st.write('3. Help them manage their time better and establish a consistent study routine.')
    st.write('4. Address any potential motivational, personal or external factors hindering their academic progress.')
    st.write('5. Provide regular advising and accountability to ensure they implement the recommended strategies effectively.')
    st.write('---')
elif student_result[prediction] == 'Graduate':
    st.subheader('Student is likely to graduate')
    st.write('---')
    st.write('Learning Strategies for Students Likely to Graduate')
    st.write('1. Reinforce effective strategies they may already be using like active reading, note-taking, self-testing, and time management.')
    st.write('2. Introduce additional strategies like concept mapping, chunking content into manageable parts, setting up a study schedule, etc.')
    st.write('3. Encourage them to reflect on and optimize their current study habits for continuous improvement.')
    st.write('4. Provide resources and support for developing advanced skills like critical thinking, research, and writing.')
    st.write('---')

targets = ['Dropout', 'Enrolled', 'Graduate']
st.subheader('Targets')
columns = ['Target']
targets_df = pd.DataFrame(targets, columns = columns)
st.write(targets_df)
st.write('---')

st.subheader('Prediction probability')
cols = ['Dropout', 'Enrolled', 'Graduate']
result = pd.DataFrame(prediction_proba, columns = cols)
st.write(result)

st.write('---')
