import csv
import torch
import pandas as pd
import torchtext
from torchtext.data import get_tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

data = pd.read_csv('iphi2802.csv',delimiter='\t')  

labels = data.columns
print(labels)

tokenizer = get_tokenizer('basic_english')
text = data['text'].tolist()
# tokenized_text = [tokenizer(sentence) for sentence in text]
tokenized_text = [' '.join(tokenizer(sentence)) for sentence in text]

# Initialize TfidfVectorizer with max_features
max_features = 1000  # Number of features (tokens) to keep
vectorizer = TfidfVectorizer(max_features=max_features)

# Fit and transform the tokenized text to obtain TF-IDF vectors
tfidf_matrix = vectorizer.fit_transform(tokenized_text)

# Get the feature names (words) from the vectorizer
feature_names = vectorizer.get_feature_names_out()

# Convert the TF-IDF matrix to a pandas DataFrame for easier manipulation
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

# Perform min-max normalization for the date_min and date_max columns
min_date_min = data['date_min'].min()
max_date_min = data['date_min'].max()
min_date_max = data['date_max'].min()
max_date_max = data['date_max'].max()

data['date_min_normalized'] = (data['date_min'] - min_date_min) / (max_date_min - min_date_min)
data['date_max_normalized'] = (data['date_max'] - min_date_max) / (max_date_max - min_date_max)

# Drop the original date_min and date_max columns if needed
# data.drop(['date_min', 'date_max'], axis=1, inplace=True)

# Print the DataFrame with normalized date_min and date_max columns
print(data)

# Convert the DataFrame to PyTorch tensors
X = torch.tensor(tfidf_df.values, dtype=torch.float)
y = torch.tensor(data[['date_min_normalized', 'date_max_normalized']].values, dtype=torch.float)

# Define the number of folds
n_splits = 5

# Initialize the KFold cross-validator
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Iterate over the folds
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    # Get the training and test data for this fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Print the shape of the training and test sets for this fold
    print(f"Fold {fold+1}:")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    print()
