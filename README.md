# Neural Network Playground with Streamlit

This project demonstrates how to build a neural network model using Streamlit, TensorFlow/Keras, and scikit-learn. It allows users to interactively choose dataset parameters, neural network architecture, and training settings through a sidebar interface. The app will generate a scatter plot of the dataset, train a neural network model, and visualize the training metrics, model summary, test accuracy, and decision boundaries.

## Requirements

To run this project, you'll need the following Python libraries:

- streamlit
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- mlxtend

You can install all dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
Alternatively, you can manually install each dependency with the following commands:
pip install streamlit
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install tensorflow
pip install mlxtend

Features:
Dataset Selection: Choose from make_classification, make_circles, or make_moons datasets.
Data Parameters: Adjust the number of samples, features, clusters, and classes.
Model Parameters: Define the number of hidden layers, neurons in each layer, activation functions, and learning rate.
Training: Train the model using the selected parameters and visualize the training metrics (loss and accuracy).
Decision Boundaries: Visualize the decision boundary for the training data.

Interactive Sidebar Options:
Dataset: Select from make_classification, make_circles, or make_moons.
Number of Samples: Choose the total number of samples for the dataset.
Number of Features: Select the number of features for classification (1-10).
Number of Hidden Layers: Define the number of hidden layers in the neural network (1-10).
Neurons per Layer: Set the number of neurons in each hidden layer.
Activation Functions: Choose the activation functions for hidden and output layers (e.g., relu, tanh, sigmoid, softmax).
Epochs: Set the number of epochs for training.
Batch Size: Select the batch size for training.
Learning Rate: Choose the learning rate for the optimizer.
Test Data Size: Set the proportion of data for testing.

Visualizations:
Dataset Plot: The scatter plot of the raw dataset.
Training Metrics: Graphs showing training and validation loss and accuracy over epochs.
Model Summary: A detailed summary of the neural network model.
Decision Boundary: Visualization of the decision boundary for the training data.

Example Use Case:
This app is useful for:
Learning: Understanding the effects of different neural network configurations on model performance.
Experimentation: Trying different datasets and hyperparameters to see how they influence training.
Visualization: Exploring decision boundaries and training metrics interactively.
