import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Set Streamlit app title
st.title('My Streamlit App')

# Read the dataset
data = pd.read_csv("DataSetProyecto.csv", sep=',')
data1 = data.drop(['Articulo', 'DI'], axis=1)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, n_init=10)
kmeans.fit(data1)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Create scatter plot of the data points
fig1 = px.scatter(data1, x='PY', y='TC', color=labels, title='Iris Cluster Plot')
fig1.add_scatter(x=centroids[:, 0], y=centroids[:, 1], mode='markers', marker=dict(symbol='x', size=10, color='red'))

# Create pairplot with color-coded points based on cluster label
fig2 = px.scatter_matrix(data, dimensions=['PY', 'TC', 'DT'], color='cluster')

# Standardize the features
scaler = StandardScaler()
data_std = scaler.fit_transform(data1)

# Create MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_std, labels, test_size=0.3, random_state=1)

# Train the classifier
mlp.fit(X_train, y_train)

# Predict on the testing data
y_pred = mlp.predict(X_test)

# Create confusion matrix and classification report
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

# Display the results
st.write('')
st.header('Prediction:')
st.write(cr)

# Show the scatter plot and pairplot
st.plotly_chart(fig1)
st.plotly_chart(fig2)

# Show the confusion matrix
st.write('Confusion Matrix:')
st.write(cm)

# Show the info in the sidebar
st.sidebar.info("Built with Streamlit")
