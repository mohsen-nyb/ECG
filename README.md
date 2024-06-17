# ECG
This project implements several deep learning architectures to classify ECG signals using a combination of Convolutional Neural Networks (CNN), Transformers, and Recurrent Neural Networks (RNN) with attention mechanism. The model is designed to handle 12-lead ECG signals and make binary predictions.


# ECG Signal Classification for ECHO Abnormality Prediction

This project implements various deep learning architectures to encode 12-lead ECG signals and predict ECHO abnormalities. The models used include CNN, RNN, CNN-RNN, CNN with Attention mechanism, and CNN with Transformer. The repository includes training and evaluation scripts, along with utilities for data handling and metrics computation. More complex models will be added. 

## Project Structure

├── data/
│ └── # Directory to store ECG datasets
├── models/
│ ├── cnn.py
│ ├── rnn.py
│ ├── cnn_rnn.py
│ ├── cnn_att_rnn.py
│ └── cnn_transformer.py
├── results/
│ └── # Directory to save training metrics and model checkpoints
├── utils.py
├── train.py
├── eval.py
└── README.md

### Model Architectures

- **CNN**: Implemented in `models/cnn.py`
- **RNN**: Implemented in `models/rnn.py`
- **CNN-RNN**: Implemented in `models/cnn_rnn.py`
- **CNN with Attention RNN**: Implemented in `models/cnn_att_rnn.py`
- **CNN with Transformer**: Implemented in `models/cnn_transformer.py`



## Usage
Preparing the Dataset
Place your 12-lead ECG data in the data/ directory. Ensure the data is preprocessed and split into training and validation sets.

Training the Model
You can train the model by running the train.py script. This script initializes the data loaders, model, loss function, and optimizer, and then trains the model for a specified number of epochs.

python train.py

Example Main Function
The main function in the train.py script allows you to select and train different models. Here is an example:
def main():
    model = ECG_CNN1D_LSTM_att()
    train(model=model, model_name='ECG_CNNID_Attention_LSTM', batch_size=32, num_epochs=30, lr=0.0001, seed=42)

if __name__ == "__main__":
    main()


Evaluating the Model
Metrics are computed and saved during training, and the best model based on validation AUC score is saved in the results/ directory. The evaluation metrics include Loss, Accuracy, Precision, Recall, F1 Score, and ROC AUC.


## Metrics
The following metrics are computed and saved during training:

Loss
Accuracy
Precision
Recall
F1 Score
ROC AUC


## Plotting Metrics
Metrics are plotted over the number of epochs and saved as PNG files in the results/ directory.

## Results
Training and validation metrics are saved in the results/ directory and plotted for visualization. The best model is also saved in this directory.



## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request. We welcome all contributions!

## License
This project is licensed under the MIT License. See the LICENSE file for details.


## Contact
For any questions or issues, please contact Mohsen at [mohsen.nayebi98@gmail.com].


