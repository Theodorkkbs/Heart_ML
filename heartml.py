# DESCRIPTION: This Application detecs if someone has a cardiovascular disease using machine learning and python


# Important libraries

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st



# Title and Sub title
st.write(""""""
         # Cardiovascular Disease Detection 
         # 1:Male 0:Female
         # thalassemia 3=normal 6=fixed 7=reversable defect
         # chest pain: starts from low which is 0 and goes up to 3
         # exercise_induced_angina 0:no 1:yes
         # resting blood sugar 120mg/dl 1:true 0:no
         """"""
         )

# display image
#image = Image.open('C:/Users/GetsomeHate/PycharmProjects/Heart_ML/heart.jpg')  remove comment and put your image here
#st.image(image, caption='Machine Learning', use_column_width=True)
# get Data
url="https://github.com/Theodorkkbs/Heart_ML/blob/main/heart.csv"
df = pd.read_csv(url,error_bad_lines=False) #df = pd.read_csv('C:/Users/GetsomeHate/PycharmProjects/Heart_ML/heart.csv') my route


st.subheader('Data Information :')
# show data in a table

st.dataframe(df)
# show statistics
st.write(df.describe())
# show data in chart
chart = st.bar_chart(df)

# splitting Data

X = df.drop('target', axis=1)
y = df['target']
# 70% training 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# get input from user
def get_user_input():

    age = st.sidebar.slider('age', 20, 100, 50)
    sex = st.sidebar.slider('sex', 0, 1, 1)
    chest_pain = st.sidebar.slider('chest_pain', 0, 3, 1)
    resting_blood_pressure = st.sidebar.slider('resting_blood_pressure', 50, 200, 80)
    cholesterol = st.sidebar.slider('cholesterol', 120, 564, 140)
    blood_sugar = st.sidebar.slider('blood_sugar', 0, 1, 0)
    ecg_results = st.sidebar.slider('ecg_results', 0, 2, 1)
    max_heart_rate = st.sidebar.slider('max_heart_rate', 50, 200, 70)
    exercise_induced_angina = st.sidebar.slider('exercise_induced_angina', 0, 1, 0)
    depression_induced_by_exercise = st.sidebar.slider('depression_induced_by_exercise', 0.0, 6.2, 1.0)
    slope_of_peak_exercise = st.sidebar.slider('slope_of_peak_exercise', 0, 2, 1)
    number_of_major_vessels = st.sidebar.slider('number_of_magor_vessels', 0, 4, 1)
    thalassemia = st.sidebar.slider('thalassemia', 0, 3, 2)

    user_data = {
        'age': age,
        'sex': sex,
        'chest_pain': chest_pain,
        'resting_blood_pressure': resting_blood_pressure,
        'cholesterol': cholesterol,
        'blood_sugar': blood_sugar,
        'ecg_results': ecg_results,
        'max_heart_rate': max_heart_rate,
        'exercise_induced_angina': exercise_induced_angina,
        'depression_induced_by_exercise': depression_induced_by_exercise,
        'slope_of_peak_exercise': slope_of_peak_exercise,
        'number_of_major_vessels': number_of_major_vessels,
        'thalassemia': thalassemia
    }

    #transform into DF
    user_df = pd.DataFrame(user_data, index=[0])
    return user_df


# store user input into variable
user_input = get_user_input()

# setting subheader and display the user input
st.subheader('User Input :')
st.write(user_input)

# Create and train model

rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)

predictions = rfc.predict(X_test)

# show metrics
st.subheader('Model Metric Accuracy :')
st.write(str(accuracy_score(y_test, predictions) * 100) + '%')

# store predictions for user input

user_pred = rfc.predict(user_input)

# set subheader to display classification

st.subheader('Classification')
st.write(user_pred)
