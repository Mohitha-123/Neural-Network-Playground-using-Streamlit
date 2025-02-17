# Neural Network Playground with Streamlit

This project is a neural network playground built with Streamlit. It allows users to interactively create and train neural networks on various synthetic datasets. The application provides a user-friendly interface for configuring dataset parameters, neural network architecture, and training parameters.

## Features

- Select from different synthetic datasets: `make_classification`, `make_circles`, `make_moons`
- Configure dataset parameters such as the number of samples, features, informative features, clusters, etc.
- Configure neural network parameters such as the number of epochs, batch size, activation functions, number of hidden layers, and neurons per layer.
- Display raw dataset scatter plot.
- Display training and validation metrics (loss and accuracy).
- Display model summary.
- Evaluate test loss and accuracy.
- Visualize decision boundary on the training data.

## Installation

To run this application, you need to have Python installed on your system. Follow the steps below to set up the environment and run the application:

1. Clone the repository:

```sh
git clone https://github.com/yourusername/neural-network-playground.git
cd neural-network-playground
```

2. Create a virtual environment and activate it:

```sh
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install the required packages:

```sh
pip install -r requirements.txt
```

## Usage

To start the Streamlit application, run the following command:

```sh
streamlit run app.py
```

This will open a new tab in your web browser with the Streamlit application.

## Configuration

Use the sidebar to configure the following parameters:

### Dataset Parameters

- **Dataset**: Select the synthetic dataset (`make_classification`, `make_circles`, `make_moons`)
- **Number of samples**: Adjust the number of samples in the dataset
- **Number of features**: Adjust the number of features (only for `make_classification`)
- **Number of informative features**: Adjust the number of informative features (only for `make_classification`)
- **Number of clusters**: Adjust the number of clusters (only for `make_classification`)

### Model Parameters

- **Number of epochs**: Adjust the number of training epochs
- **Batch size**: Adjust the batch size for training
- **Activation function for hidden layers**: Select the activation function for hidden layers (`tanh`, `relu`)
- **Activation function for output layer**: Select the activation function for the output layer (`sigmoid`, `softmax`)
- **Number of hidden layers**: Adjust the number of hidden layers
- **Number of neurons in hidden layers**: Adjust the number of neurons in each hidden layer
- **Test Data Size**: Adjust the size of the test data
- **Learning Rate**: Adjust the learning rate

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- [Streamlit](https://streamlit.io/) for providing an easy-to-use framework for building interactive web applications.
- [scikit-learn](https://scikit-learn.org/) for providing tools for data generation and preprocessing.
- [TensorFlow](https://www.tensorflow.org/) for providing tools for building and training neural networks.
- [mlxtend](http://rasbt.github.io/mlxtend/) for providing tools for plotting decision regions.