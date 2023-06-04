import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load the data
df = pd.read_excel('/Users/jmlz_rp/Documents/SistemasIA/Parcial-Iris/iris3Dplot/Proyectofinal/ProyectoFinalMLP-Julian-JoseM/A-wos_scopus.xlsx')

st.title('MLP Scientometric Recommendation')

# Display the original data
st.subheader('Original Data')
st.write(df)

# Describe the data
st.subheader('Data Description')
st.write(df.describe(include='all').T)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, n_init=10)
kmeans.fit(df[['PY', 'TC']])
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Create scatter plot of the data points
fig1 = px.scatter(df, x='PY', y='TC', color=labels, title='TC years')
fig1.add_scatter(x=centroids[:, 0], y=centroids[:, 1], mode='markers', marker=dict(symbol='x', size=10, color='red'))

# Create pairplot with color-coded points based on cluster label
fig2 = px.scatter_matrix(df, dimensions=['PY', 'TC', 'DT'], color=labels)

# Standardize the features
scaler = StandardScaler()
data_std = scaler.fit_transform(df[['PY', 'TC']])

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

# Show the scatter plot and pairplot
st.plotly_chart(fig1)
st.plotly_chart(fig2)

# Show the confusion matrix
st.subheader('Confusion Matrix')
st.write(cm)

# Show the classification report
st.subheader('Classification Report')
st.write(cr)

# Conclusion
st.header('Conclusion')
st.markdown('In this Streamlit app, we have explored the original data, performed KMeans clustering, and created visualizations such as scatter plots, pair plots, and a confusion matrix. Additionally, we trained an MLP classifier on the dataset and evaluated its performance using a classification report. Streamlit makes it easy to build interactive and informative apps for data exploration and model evaluation.')

# Add a file uploader to upload an image
st.sidebar.title("Upload Seed from WebOfScience .txt")
uploaded_file = st.sidebar.file_uploader("", type=["txt"])


import streamlit as st
import pandas as pd
import plotly.express as px

# Load the data
original = pd.read_excel('/Users/jmlz_rp/Documents/SistemasIA/Parcial-Iris/iris3Dplot/Proyectofinal/ProyectoFinalMLP-Julian-JoseM/A-wos_scopus.xlsx')
df = pd.read_excel('/Users/jmlz_rp/Documents/SistemasIA/Parcial-Iris/iris3Dplot/Proyectofinal/ProyectoFinalMLP-Julian-JoseM/A-wos_scopus.xlsx')

st.title('EDA')

# Transformations according to Quartiles
st.subheader('TC Description')
st.write(df['TC'].describe())

# Count CR occurrences
df['CR-count'] = df['CR'].apply(lambda x: x.count(';') if isinstance(x, str) else 0)

# Normalize CR count
df['CR-norm'] = df['CR-count'].apply(lambda x: '1' if x > df['CR-count'].min() and x < df['CR-count'].quantile(.25) else ('2' if x > df['CR-count'].quantile(.25) and x < df['CR-count'].quantile(.5) else ('3')))

# Normalize TC
df['TC-norm'] = df['TC'].apply(lambda x: '1' if x >= 0 and x <= 300 else '2' if x > 300 and x < 500 else ('3' if x > 500 else x))

# Normalize PY
df['PY-norm'] = df['PY'].apply(lambda x: '1' if x > df['PY'].min() and x < df['PY'].quantile(.25) else ('2' if x > df['PY'].quantile(.25) and x < df['PY'].quantile(.50) else ('3')))

# Normalize DT
df['DT-norm'] = df['DT'].apply(lambda x: '3' if x == 'ARTICLE' else ('2' if x == 'CONFERENCE PAPER' else ('1')))

st.subheader('Norm Data')
st.write(df.loc[:,['PY-norm', 'CR-norm','TC-norm','DT-norm']].T)

# Plot box plots
fig1 = px.box(df, y=['PY-norm', 'CR-norm','TC-norm','DT-norm','CR-norm'], points="all", color=labels)
st.plotly_chart(fig1)

