# ECG
This project implements several deep learning architectures to classify ECG signals using a combination of Convolutional Neural Networks (CNN), Transformers, and Recurrent Neural Networks (RNN) with attention mechanism. The model is designed to handle 12-lead ECG signals and make binary predictions.


# ECG Signal Classification for Abnormality Prediction

This project implements various deep learning architectures to encode 12-lead ECG signals and predict ECHO abnormalities. The models used include CNN, RNN, CNN-RNN, CNN with Attention mechanism, and CNN with Transformer. The repository includes training and evaluation scripts, along with utilities for data handling and metrics computation. More complex models will be added. 


### Model Architectures

- **CNN**: Implemented in `models/cnn.py`
  This model is a 1D Convolutional Neural Network (CNN) designed for classifying 12-lead ECG signals, consisting of multiple convolutional layers followed by fully connected layers, with dropout regularization applied at each layer.
  
- **RNN**: Implemented in `models/rnn.py`
  This model is a Recurrent Neural Network (RNN), specifically a Long Short-Term Memory (LSTM) network, designed for classifying 12-lead ECG signals. It processes the input signals through LSTM layers to capture temporal dependencies and then applies fully connected layers for classification, with dropout regularization applied at the fully connected layer.
  

- **CNN-RNN**: Implemented in `models/cnn_rnn.py`
  This model is a combination of a 1D Convolutional Neural Network (CNN) and a Long Short-Term Memory (LSTM) layer, designed for classifying 12-lead ECG signals. It applies convolutional layers to extract features from the input signals, followed by an LSTM layer to capture temporal dependencies, with dropout regularization applied at each layer.
  
- **CNN with Attention RNN**: Implemented in `models/cnn_att_rnn.py`
  This model combines a 1D Convolutional Neural Network (CNN) with a Long Short-Term Memory (LSTM) layer and an attention mechanism. It first applies convolutional layers to extract features from the input signals, then passes the output through an LSTM layer to capture temporal dependencies, and finally utilizes an attention mechanism to focus on relevant parts of the sequence for classification.
  
- **CNN with Transformer**: Implemented in `models/cnn_transformer.py`
  This model combines a 1D Convolutional Neural Network (CNN) with a Transformer architecture. It first applies convolutional layers to extract features from the input signals and then processes the output through a Transformer encoder, followed by fully connected layers for classification.



## Usage
Preparing the Dataset
Place your 12-lead ECG data in the data/ directory. Ensure the data is preprocessed and split into training and validation sets.

## Training the Model
You can train the model by running the train.py script. This script initializes the data loaders, model, loss function, and optimizer, and then trains the model for a specified number of epochs. The main function in the train.py script allows you to select and train different models


## Evaluating the Model
Metrics are computed and saved during training, and the best model based on validation AUC score is saved in the model_name/ directory. The evaluation metrics include Loss, Accuracy, Precision, Recall, F1 Score, and ROC AUC.


## Metrics
The following metrics are computed and saved during training:

Loss
Accuracy
Precision
Recall
F1 Score
ROC AUC


## Plotting Metrics
Metrics are plotted over the number of epochs and saved as PNG files in the model_name/ directory.

## Results
Training and validation metrics are saved in the model_name/ directory and plotted for visualization. The best model is also saved in this directory.




