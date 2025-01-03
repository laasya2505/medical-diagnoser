import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
data = pd.read_csv('/content/medical_data.csv')
data.head()
#lstm
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(data['Patient_Problem'])

sequences = tokenizer.texts_to_sequences(data['Patient_Problem'])

max_length = max(len(x) for x in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Encoding the labels
label_encoder_disease = LabelEncoder()
label_encoder_prescription = LabelEncoder()

disease_labels = label_encoder_disease.fit_transform(data['Disease'])
prescription_labels = label_encoder_prescription.fit_transform(data['Prescription'])

# Converting labels to categorical
disease_labels_categorical = to_categorical(disease_labels)
prescription_labels_categorical = to_categorical(prescription_labels)

Y = np.hstack((disease_labels_categorical, prescription_labels_categorical))

input_layer = Input(shape=(max_length,))

embedding = Embedding(input_dim=5000, output_dim=64)(input_layer)
lstm_layer = LSTM(64)(embedding)

disease_output = Dense(len(label_encoder_disease.classes_), activation='softmax',
name='disease_output')(lstm_layer)

prescription_output = Dense(len(label_encoder_prescription.classes_),
activation='softmax', name='prescription_output')(lstm_layer)

model = Model(inputs=input_layer, outputs=[disease_output, prescription_output])

model.compile(
    loss={'disease_output': 'categorical_crossentropy',
    'prescription_output': 'categorical_crossentropy'},
    optimizer='adam',
    metrics={'disease_output': ['accuracy'], 'prescription_output': ['accuracy']}
)

model.summary()
model.fit(padded_sequences, {'disease_output': disease_labels_categorical, 'prescription_output':
      prescription_labels_categorical}, epochs=100, batch_size=32)
# Evaluate LSTM model
eval_lstm = model.evaluate(X_test, {'disease_output': Y_test_disease, 'prescription_output': Y_test_prescription})
acc_disease_lstm = eval_lstm[3]
acc_prescription_lstm = eval_lstm[4]
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, Y_train_disease, Y_test_disease, Y_train_prescription, Y_test_prescription = train_test_split(
    padded_sequences,
    disease_labels_categorical,
    prescription_labels_categorical,
    test_size=0.2,
    random_state=42
)
# Verify shapes
print(X_test.shape)  # Should be (num_samples, max_length)
print(Y_test_disease.shape)  # Should match the output of disease_output layer
print(Y_test_prescription.shape)  # Should match the output of prescription_output layer
#GRU
from tensorflow.keras.layers import GRU, Embedding, Dense, Input
from tensorflow.keras.models import Model

input_layer = Input(shape=(max_length,))
embedding = Embedding(input_dim=5000, output_dim=64)(input_layer)
gru_layer = GRU(64)(embedding)

disease_output = Dense(len(label_encoder_disease.classes_), activation='softmax', name='disease_output')(gru_layer)
prescription_output = Dense(len(label_encoder_prescription.classes_), activation='softmax', name='prescription_output')(gru_layer)

gru_model = Model(inputs=input_layer, outputs=[disease_output, prescription_output])
gru_model.compile(
    optimizer='adam',
    loss={
        'disease_output': 'categorical_crossentropy',
        'prescription_output': 'categorical_crossentropy'
    },
    metrics={
        'disease_output': ['accuracy'],
        'prescription_output': ['accuracy']
    }
)

# Evaluate the GRU model
eval_gru = gru_model.evaluate(
    X_test,
    {'disease_output': Y_test_disease, 'prescription_output': Y_test_prescription}
)

# Extract and print accuracy metrics for GRU
disease_acc_gru = eval_gru[3]  # Disease accuracy (3rd element)
prescription_acc_gru = eval_gru[4]  # Prescription accuracy (4th element)

print(f"GRU Model Accuracy for Disease Prediction: {disease_acc_gru:.4f}")
print(f"GRU Model Accuracy for Prescription Prediction: {prescription_acc_gru:.4f}")

predictions = gru_model.predict(X_test[:5])  # Predict a few samples
print(predictions)
# Evaluate GRU model
eval_gru = gru_model.evaluate(X_test, {'disease_output': Y_test_disease, 'prescription_output': Y_test_prescription})
acc_disease_gru = eval_gru[3]
acc_prescription_gru = eval_gru[4]
#BIDIRECTIONAL RNN
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, SimpleRNN

# Define input layer
input_layer = Input(shape=(max_length,))

# Embedding layer
embedding_layer = Embedding(input_dim=5000, output_dim=64, input_length=max_length)(input_layer)

# Bidirectional RNN layer
bidirectional_rnn = Bidirectional(SimpleRNN(128, return_sequences=False))(embedding_layer)

# Dense layer for shared features
dense_layer = Dense(128, activation='relu')(bidirectional_rnn)

# Separate output layers for disease and prescription
disease_output = Dense(len(label_encoder_disease.classes_), activation='softmax', name='disease_output')(dense_layer)
prescription_output = Dense(len(label_encoder_prescription.classes_), activation='softmax', name='prescription_output')(dense_layer)

# Define Bidirectional RNN model with multiple outputs
bidir_rnn_model = Model(inputs=input_layer, outputs=[disease_output, prescription_output])

# Compile the model
# Compile the Bidirectional RNN model with separate metrics for each output
bidir_rnn_model.compile(
    optimizer='adam',
    loss={
        'disease_output': 'categorical_crossentropy',
        'prescription_output': 'categorical_crossentropy'
    },
    metrics={
        'disease_output': ['accuracy'],
        'prescription_output': ['accuracy']
    }
)

# Train the Bidirectional RNN model
history_bidir_rnn = bidir_rnn_model.fit(
    padded_sequences,
    {
        'disease_output': disease_labels_categorical,
        'prescription_output': prescription_labels_categorical
    },
    epochs=50,
    batch_size=32,
    validation_split=0.2
)
# Evaluate the Bidirectional RNN model
eval_results = bidir_rnn_model.evaluate(
    padded_sequences,
    {'disease_output': disease_labels_categorical, 'prescription_output': prescription_labels_categorical}
)

# The eval_results list might have the following structure:
# [total_loss, disease_output_loss, prescription_output_loss, disease_output_accuracy, prescription_output_accuracy]
# Adjust the unpacking based on the actual structure

# Unpack the results
# (loss for each output, accuracy for each output, and potentially other metrics)
loss_disease_bidir_rnn, loss_prescription_bidir_rnn, acc_disease_bidir_rnn, acc_prescription_bidir_rnn, _  = eval_results


# Print the evaluation results
print(f"Loss (Disease): {loss_disease_bidir_rnn:.4f}")
print(f"Loss (Prescription): {loss_prescription_bidir_rnn:.4f}")
print(f"Accuracy (Disease): {acc_disease_bidir_rnn:.4f}")
print(f"Accuracy (Prescription): {acc_prescription_bidir_rnn:.4f}")
# Evaluate BiRNN model
eval_birnn = bidir_rnn_model.evaluate(X_test, {'disease_output': Y_test_disease, 'prescription_output': Y_test_prescription})
acc_disease_birnn = eval_birnn[3]
acc_prescription_birnn = eval_birnn[4]
#BERT
pip install transformers
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

# Load and preprocess the dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the input text (Patient_Problem)
inputs = tokenizer(list(data['Patient_Problem']), padding=True, truncation=True, max_length=128, return_tensors="pt")

# Encode labels for Disease and Prescription
label_disease = torch.tensor(pd.factorize(data['Disease'])[0])
label_prescription = torch.tensor(pd.factorize(data['Prescription'])[0])

# Create DataLoader for training
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], label_disease, label_prescription)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define the BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(data['Disease'].unique()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set the optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
total_steps = len(loader) * 10  # Assume 10 epochs
warmup_steps = int(0.1 * total_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

# Early stopping setup
best_val_loss = float('inf')
patience = 3
patience_counter = 0

# Train the model
model.train()
for epoch in range(30):  # Train for more epochs
    for batch in loader:
        b_input_ids, b_attention_mask, b_labels = [x.to(device) for x in batch[:3]]
        outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update the learning rate
        optimizer.zero_grad()

    # Early stopping based on validation loss
    if loss < best_val_loss:
        best_val_loss = loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter > patience:
            print("Early stopping")
            break

# Evaluate the model (example for disease prediction)
model.eval()
predictions_disease = []
predictions_prescription = []

for batch in loader:
    b_input_ids, b_attention_mask, b_labels_disease, b_labels_prescription = [x.to(device) for x in batch]
    with torch.no_grad():
        outputs = model(b_input_ids, attention_mask=b_attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        predictions_disease.extend(preds.cpu().numpy())

# Convert test labels to numpy arrays
Y_test_disease = label_disease.numpy()
Y_test_prescription = label_prescription.numpy()

# Calculate accuracy for disease prediction
acc_disease_bert = accuracy_score(Y_test_disease, predictions_disease)
print(f"BERT Model Accuracy for Disease Prediction: {acc_disease_bert:.4f}")
# Assuming the model and DataLoader are already defined

# Initialize lists to collect predictions
predictions_disease = []
predictions_prescription = []

# Set model to evaluation mode
model.eval()

# Iterate over batches in the DataLoader
for batch in loader:
    b_input_ids, b_attention_mask, b_labels_disease, b_labels_prescription = [x.to(device) for x in batch]

    with torch.no_grad():
        outputs = model(b_input_ids, attention_mask=b_attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        predictions_disease.extend(preds.cpu().numpy())  # Collect disease predictions
        predictions_prescription.extend(preds.cpu().numpy())  # Collect prescription predictions

# Ensure predictions length matches the number of labels
assert len(predictions_disease) == len(label_disease), f"Prediction length: {len(predictions_disease)}, Label length: {len(label_disease)}"

# Calculate accuracy for disease prediction
acc_disease_bert = accuracy_score(label_disease.numpy(), predictions_disease)
acc_prescription_bert = accuracy_score(label_prescription.numpy(), predictions_prescription)

# Print results
print(f"BERT Model Accuracy for Disease Prediction: {acc_disease_bert:.4f}")
print(f"BERT Model Accuracy for Prescription Prediction: {acc_prescription_bert:.4f}")
#comparision of models
import pandas as pd

# Assuming you have the accuracy values for all models (LSTM, GRU, BiRNN, BERT)
comparison_df = pd.DataFrame({
    'Model': ['LSTM', 'Bidirectional RNN', 'BERT'],
    'Accuracy (Disease)': [acc_disease_lstm, acc_disease_birnn, acc_disease_bert],
    'Accuracy (Prescription)': [acc_prescription_lstm, acc_prescription_birnn, acc_prescription_bert]
})

# Display the comparison table
print(comparison_df)
# Visualize the comparison (Bar plot)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
comparison_df.set_index('Model')[['Accuracy (Disease)', 'Accuracy (Prescription)']].plot(kind='bar', stacked=True)
plt.title("Model Comparison: Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Models")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
#prediction with LSTM model
def make_predictions(patient_problems):
    # Iterate through each patient problem
    for i, problem in enumerate(patient_problems):
        # Preprocess the input
        sequence = tokenizer.texts_to_sequences([problem])
        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

        # Make prediction
        prediction = model.predict(padded_sequence, verbose=0)

        # Decode the prediction
        disease_index = np.argmax(prediction[0], axis=1)[0]
        prescription_index = np.argmax(prediction[1], axis=1)[0]

        disease_predicted = label_encoder_disease.inverse_transform([disease_index])[0]
        prescription_predicted = label_encoder_prescription.inverse_transform([prescription_index])[0]

        # Print the results for each case
        print(f"Case {i+1}:")
        print(f"Patient Problem: {problem}")
        print(f"Predicted Disease: {disease_predicted}")
        print(f"Suggested Prescription: {prescription_predicted}")
        print("-" * 50)  # Separator for readability


# Example inputs for multiple cases
patient_inputs = [
    "I've experienced a loss of appetite and don't enjoy food anymore.",
    "I have a severe headache and blurred vision.",
    "I feel shortness of breath and chest pain.",
    "My joints are swollen and I feel constant fatigue."
]

# Make predictions for multiple cases
make_predictions(patient_inputs)
