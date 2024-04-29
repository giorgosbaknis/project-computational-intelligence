import csv
import torch
import pandas as pd
import torchtext
from torchtext.data import get_tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('iphi2802.csv',delimiter='\t')  

labels = data.columns
print(labels)

tokenizer = get_tokenizer('basic_english')
text = data['text'].tolist()

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
