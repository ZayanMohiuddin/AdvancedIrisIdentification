# S10.1: Copy this code cell in 'iris_app.py' using the Sublime text editor. You have already created this ML model in the previous class(es).

# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")
# Loading the dataset.
iris_df = pd.read_csv("iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating a model for Support Vector classification to classify the flower types into labels '0', '1', and '2'.

# Creating features and target DataFrames.
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

# Splitting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Creating the SVC model and storing the accuracy score in a variable 'score'.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
score = svc_model.score(X_train, y_train)

lr = LogisticRegression()
lr.fit(X_train, y_train)
score_lr = lr.score(X_train, y_train)

rcf = RandomForestClassifier()
rcf.fit(X_train, y_train)
rcf_score = rcf.score(X_train, y_train)
# S10.2: Perform this activity in Sublime editor after adding the above code. 
# Create a function 'prediction()' that accepts 'SepalLength', 'SepalWidth', 'PetalLength' and 'PetalWidth' as inputs and returns the species name.
@st.cache_data
def prediction(SepalLength, SepalWidth, PetalLength, PetalWidth, classifiers):
  species = classifiers.predict([[SepalLength, SepalWidth, PetalLength, PetalWidth]])
  species = species[0]
  if species == 0:
    return "Iris-setosa"
  elif species == 1:
    return "Iris-virginica"
  else:
    return "Iris-versicolor"
# S10.3: Perform this activity in Sublime editor after adding the above code. 
# Add title widget
st.title("Iris Flower Species Prediction App")  

# Add 4 sliders and store the value returned by them in 4 separate variables.
st.sidebar.subheader("Variable Adjusters")
s_length = st.sidebar.slider("Sepal Length", float(iris_df["SepalLengthCm"].min()), float(iris_df["SepalLengthCm"].max()))
s_width = st.sidebar.slider("Sepal Width", float(iris_df["SepalWidthCm"].min()), float(iris_df["SepalWidthCm"].max()))
p_length = st.sidebar.slider("Petal Length", float(iris_df["PetalLengthCm"].min()), float(iris_df["PetalLengthCm"].max()))
p_width = st.sidebar.slider("Petal Width", float(iris_df["PetalWidthCm"].min()), float(iris_df["PetalWidthCm"].max()))

classifiers = st.sidebar.selectbox("Model Type", ("Support Vector Classifier", "Logistic Regression", "Random Forest Classifier"))
# When 'Predict' button is pushed, the 'prediction()' function must be called 
# and the value returned by it must be stored in a variable, say 'species_type'. 
# Print the value of 'species_type' and 'score' variable using the 'st.write()' function.
if st.button("Predict"):
  if classifiers == "Support Vector Classifier":
    species_type = prediction(s_length, s_width, p_length, p_width, svc_model)
    st.write("Species predicted:", species_type)
    st.write("Accuracy score of this model is:", score)

  elif classifiers == "Logistic Regression":
    species_type = prediction(s_length, s_width, p_length, p_width, lr)
    st.write("Species predicted:", species_type)
    st.write("Accuracy score of this model is:", score_lr)

  else:
    species_type = prediction(s_length, s_width, p_length, p_width, rcf)
    st.write("Species predicted:", species_type)
    st.write("Accuracy score of this model is:", rcf_score)