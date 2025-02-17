
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from mlxtend.plotting import plot_decision_regions
import io

st.title("Neural Network Playground with Streamlit")

st.sidebar.header("Data and Model Parameters")

dataset_option = st.sidebar.selectbox('Select Dataset', ('make_classification', 'make_circles', 'make_moons'))
n_samples = st.sidebar.slider('Number of samples', min_value=600, max_value=2000, value=1000, step=200)
n_features = st.sidebar.slider('Number of features', min_value=1, max_value=10, value=4, step=1) if dataset_option == "make_classification" else 2
# n_informative = st.sidebar.slider('Number of informative features', min_value=1, max_value=n_features, value=1, step=1) if dataset_option == "make_classification" else 2
n_informative = n_features
n_redundant = 0
n_classes = 2
class_sep = 1.5
n_clusters = st.sidebar.slider('Number of clusters', min_value=1, max_value=3, value=1, step=1) if dataset_option == "make_classification" else 0
noise = st.sidebar.slider('Noise', min_value=0.1, max_value=0.2, value=0.1, step=0.02) if dataset_option != "make_classification" else 0
factor = st.sidebar.slider('Factor', min_value=0.1, max_value=0.7, value=0.4, step=0.1) if dataset_option == "make_circles" else 0

# epochs = st.sidebar.slider('Number of epochs', min_value=0, max_value=200, value=50, step=50)
epochs = st.sidebar.slider('Number of epochs', min_value=25, max_value=100, value=25, step=25)
batch_size = st.sidebar.slider('Batch size', min_value=10, max_value=100, value=50, step=10)
activation_hidden = st.sidebar.selectbox('Activation function for hidden layers', ['tanh', 'relu','sigmoid'])
activation_output = st.sidebar.selectbox('Activation function for output layer', ['sigmoid', 'softmax'])
n_hidden_layers = st.sidebar.slider('Number of hidden layers', min_value=1, max_value=10, value=3, step=1)

neurons_per_layer = []
for i in range(n_hidden_layers):
    neurons = st.sidebar.slider(f'Number of neurons in hidden layer {i+1}', min_value=1, max_value=8, value=4, step=1)
    neurons_per_layer.append(neurons)

test_data = st.sidebar.slider("Test Data Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
l_rate = st.sidebar.selectbox("Learning Rate", options=[0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],index=5)

if st.sidebar.button('Submit'):
    st.warning("🤖 Neural networks need a moment to think... Wait for 30 sec or go save the world!")
    
    if dataset_option == 'make_classification':
        fv, cv = make_classification(n_samples=n_samples, n_features=n_features,n_informative=n_informative, n_redundant=n_redundant,
                                    n_repeated=0, n_classes=n_classes,class_sep=class_sep,n_clusters_per_class=n_clusters,
                                    random_state=10)
    elif dataset_option == 'make_circles':
        fv, cv = make_circles(n_samples=n_samples, noise= noise, factor= factor, random_state=10)
    elif dataset_option == 'make_moons':
        fv, cv = make_moons(n_samples=n_samples, noise= noise, random_state=10)
    
    st.write("### Raw Dataset")
    fig, ax = plt.subplots()
    sns.scatterplot(x=fv[:, 0], y=fv[:, 1], hue=cv, ax=ax)
    plt.title("Scatter Plot of the Data")
    st.pyplot(fig)

    x_train, x_test, y_train, y_test = train_test_split(fv, cv, test_size=test_data, random_state=10, stratify=cv)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

#   Apply PCA if more than 2 feature
    if x_train_scaled.shape[1] > 2:
        pca = PCA(n_components=2)
        x_train_pca = pca.fit_transform(x_train_scaled)
        x_test_pca = pca.transform(x_test_scaled)
    else:
        x_train_pca = x_train_scaled
        x_test_pca = x_test_scaled

    model = Sequential()
    model.add(InputLayer(input_shape=(x_train_pca.shape[1],)))
    for neurons in neurons_per_layer:
        model.add(Dense(units=neurons, activation=activation_hidden))
    model.add(Dense(units=1 if n_classes == 2 else n_classes, activation=activation_output))

    loss_fn = "binary_crossentropy" if n_classes == 2 else "sparse_categorical_crossentropy"
    model.compile(optimizer=SGD(learning_rate=l_rate), loss=loss_fn, metrics=["accuracy"])

    history = model.fit(x_train_pca, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

    # Save the trained model
    import pickle
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    st.write("### Training and Validation Metrics")
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(history.history['loss'], label='Train Loss')
    axs[0].plot(history.history['val_loss'], label='Validation Loss')
    axs[0].set_title('Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(history.history['accuracy'], label='Train Accuracy')
    axs[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axs[1].set_title('Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    st.pyplot(fig)

    # st.write("### Model Summary")
    # model_summary_str = io.StringIO()
    # model.summary(print_fn=lambda x: model_summary_str.write(x + '\n'))
    # st.text(model_summary_str.getvalue())

    test_loss, test_accuracy = model.evaluate(x_test_pca, y_test, verbose=1)
    st.write(f"### Test Loss: {test_loss:.4f}")
    st.write(f"### Test Accuracy: {test_accuracy:.4f}")

    class KerasClassifierWrapper:
        def __init__(self, model):
            self.model = model
        def predict(self, X):
            return self.model.predict(X).argmax(axis=1) if n_classes > 2 else (self.model.predict(X) > 0.5).astype(int)

    wrapper = KerasClassifierWrapper(model)
    
    st.write("### Decision Boundary - Training Data")
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_decision_regions(x_train_pca, y_train.astype(int), clf=wrapper, zoom_factor=4, ax=ax)
    st.pyplot(fig)
