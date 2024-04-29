import csv
import torch
import pandas as pd
import torchtext
from torchtext.data import get_tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Define the neural network model
# class NeuralNet(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(NeuralNet, self).__init__()
#         self.hidden = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.output = nn.Linear(hidden_size, output_size)
        
#     def forward(self, x):
#         x = self.relu(self.hidden(x))
#         x = self.output(x)
#         return x

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.hidden2 = nn.Linear(hidden_size, hidden_size//2) # Second hidden layer
        self.relu2 = nn.ReLU()
        self.hidden3 = nn.Linear(hidden_size//2, hidden_size//4) # Third hidden layer
        self.relu3 = nn.ReLU()
        self.output = nn.Linear(hidden_size//4, output_size)
        
    def forward(self, x):
        x = self.relu1(self.hidden1(x))
        x = self.relu2(self.hidden2(x)) # Pass through the second hidden layer
        x = self.relu3(self.hidden3(x)) # Pass through the third hidden layer
        x = self.output(x)
        return x

# Load data
data = pd.read_csv('../iphi2802.csv', delimiter='\t')

# Tokenize text
tokenizer = get_tokenizer('basic_english')
text = data['text'].tolist()
tokenized_text = [' '.join(tokenizer(sentence)) for sentence in text]

# TF-IDF Vectorization
max_features = 1000  # Number of features (tokens) to keep
vectorizer = TfidfVectorizer(max_features=max_features)
tfidf_matrix = vectorizer.fit_transform(tokenized_text)
feature_names = vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

# Normalize date_min and date_max columns
min_date_min = data['date_min'].min()
max_date_min = data['date_min'].max()
min_date_max = data['date_max'].min()
max_date_max = data['date_max'].max()

data['date_min_normalized'] = (data['date_min'] - min_date_min) / (max_date_min - min_date_min)
data['date_max_normalized'] = (data['date_max'] - min_date_max) / (max_date_max - min_date_max)

# Convert data to PyTorch tensors
X = torch.tensor(tfidf_df.values, dtype=torch.float)
y = torch.tensor(data[['date_min_normalized']].values, dtype=torch.float)

# Define hyperparameters
learning_rate = 0.1
m= 0.6
epochs = 1000
hidden_units = 100  # Number of neurons in the hidden layer
input_units = max_features
output_units = 1  # date_min_normalized 

# Define the model, loss function, and optimizer
model = NeuralNet(input_units, hidden_units, output_units)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=m)

# Define the number of folds
n_splits = 5

# Initialize the KFold cross-validator
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Define early stopping parameters
patience = 10 # Number of epochs to wait for improvement
min_delta = 0.001 # Minimum change in the monitored metric to qualify as an improvement
loss_values = []
# Iterate over the folds
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    # Get the training and test data for this fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_train)

        # Calculate RMSE loss
        loss = torch.sqrt(criterion(outputs, y_train))
        
        # Store the loss value
        loss_values.append(loss.item())
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
         
        # Print progress every 100 epochs
        if epoch % 100 == 0:
            print(f'Fold {fold+1}, Epoch [{epoch+1}/{epochs}], Loss RMSE: {loss.item():.4f} ')

    # Once training is done for this fold, evaluate on test set
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = torch.sqrt(criterion(test_outputs, y_test))
        test_rmse = test_loss
        print(f'Fold {fold+1}, Test RMSE: {test_rmse.item():.4f}')

    # Check for improvement
    if test_loss < best_val_loss - min_delta:
        best_val_loss = test_loss
        epochs_no_improve = 0
        # Save the model weights
        torch.save(model.state_dict(), f'best_model_weights_fold_{fold+1}.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve > patience:
            print(f'Early stopping at epoch {epoch+1} for fold {fold+1}')
            break


# Plot the loss values
plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.show()