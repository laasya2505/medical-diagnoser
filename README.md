# medical-diagnoser
This repository contains a comprehensive implementation of multi-task learning models for predicting diseases and prescriptions from patient-reported problems. It leverages advanced natural language processing (NLP) techniques and various neural network architectures, including LSTM, GRU, Bidirectional RNNs, and BERT.

Project Overview
The project aims to develop a multi-task deep learning framework that simultaneously predicts diseases and prescriptions based on patient problem descriptions. Each model architecture is evaluated for its performance, and the results are compared to identify the most effective approach.

Key Features
Multi-Task Learning:

Separate outputs for disease and prescription predictions.
Shared embeddings and learned features to optimize task performance.
Model Architectures:

LSTM (Long Short-Term Memory): Captures sequential dependencies in patient problem descriptions.
GRU (Gated Recurrent Unit): Offers computational efficiency while maintaining performance.
Bidirectional RNN: Enhances context understanding by processing sequences in both directions.
BERT (Bidirectional Encoder Representations from Transformers): Pre-trained transformer-based model fine-tuned for text classification.
Data Preprocessing:

Tokenization and padding for text inputs.
Label encoding and conversion to categorical outputs for multi-class classification.
Evaluation:

Metrics: Accuracy for disease and prescription predictions.
Visualization: Bar plots for comparative performance analysis.
Predictive Functionality:

User-friendly interface for making predictions on new patient problems using the trained LSTM model.
Dataset
The dataset comprises three key columns:

Patient_Problem: Text descriptions of patient-reported problems.
Disease: Diagnosed diseases corresponding to the problems.
Prescription: Prescriptions recommended for the diseases.
Implementation Details
Preprocessing
Text data is tokenized using Keras Tokenizer, and sequences are padded to ensure uniform input length.
Labels for Disease and Prescription are encoded using LabelEncoder and converted to categorical format for multi-class classification.
Models
LSTM: Captures temporal relationships in the input text.
GRU: Similar to LSTM but with fewer parameters, making it computationally efficient.
Bidirectional RNN: Processes the sequence in both forward and backward directions for enhanced context understanding.
BERT: Fine-tuned transformer-based model for state-of-the-art performance.
Evaluation
All models are evaluated on a test dataset, with accuracy metrics recorded for both tasks.
Comparative performance is visualized using bar plots for better insight into model effectiveness.
Visualization
A bar chart illustrates the comparative accuracy of different models for disease and prescription predictions.

Prediction
The LSTM model is used for live predictions.
Input: Patient problem descriptions.
Output: Predicted disease and prescription labels.
