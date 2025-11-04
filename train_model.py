import pandas as pd
from sklearn.model_selection import train_test_split # Used for 70-15-15. Splitting data.
from map_tags import extract_all_features

# NOTE: Put into functions (def).

processedData = pd.read_csv("processed_data.csv") # Loads data extracted from map_tags.py

#Task #1: 70-15-15 Validation Testing - Splits all the words from the ValidatedDataSet file.
# 70% becomes data training, 15% for validation, and 15% for testing.
feature_dicts = processedData['word'].apply(extract_all_features) # Extracts each feature from each word on the file.
X = pd.DataFrame(feature_dicts.tolist()) # Puts into panda dictionary (Hashmap in Java) and into a list.
y = processedData['three_class_label'] # Grabs the categories.

#Splitting:
# First split: 70% train, 30% temp. Split given by sklearn.
# test_size = Percentage to be split.
# random_state = Says how "shuffled" it should be. Can be any positive integer.
# stratify = Balances each classification, using "y" means it proportions it to 3 classifications (ENG-FIL-OTH).

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=50, stratify=y)

# Second split: Split temp into 15% val, 15% test (50-50 of the 30%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=50, stratify=y_temp)

# Print test:
print("PHASE 3.1: file 'train_model':")
print(f"Extracted {len(X.columns)} features per word")
print(y.value_counts()) #Thanks to pandas.
print()
print("Splitting Data to 70-15-15: ")
print(f" Training set:   {len(X_train)} samples")
print(f" Validation set: {len(X_val)} samples")
print(f" Test set:       {len(X_test)} samples")

#Note: Use print(f" "), makes it faster instead of doing the Java way.

# OPTION A: Naive Bayes
#   - Faster, think of it as the "Greedy BFS" of MCO1. Faster yet
#   - Study tomorrow.

# OPTION B: Decision Tree Classifier