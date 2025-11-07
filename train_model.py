import pickle # Used to save files
from sklearn.model_selection import train_test_split # Used for 70-15-15. Splitting data.
from sklearn.naive_bayes import MultinomialNB # CHANGED FROM GAUSSIAN.
from sklearn.metrics import classification_report, confusion_matrix # IS THIS ALLOWED?! IT FEELS TOO EASY ALREADY!

# Task #1: Get data
with open('phase2_output/feature_matrix.pkl', 'rb') as f:
    X = pickle.load(f)

with open('phase2_output/labels.pkl', 'rb') as f:
    y = pickle.load(f)

with open('phase2_output/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

#Task #2: 70-15-15 Validation Testing - Splits all the words from the ValidatedDataSet file.

#Splitting:
# test_size = Percentage to be split.
# random_state = Says how "shuffled" it should be. Can be any positive integer.
# stratify = Balances each classification, using "y" means it proportions it to 3 classifications (ENG-FIL-OTH).

# First split: 70% train, 30% temp. Split given by sklearn.
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=50, stratify=y)

# Second split: Split temp into 15% val, 15% test (50-50 of the 30%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=50, stratify=y_temp)

# Print test:
print("PHASE 3.1: file 'train_model':")
print(f"Extracted {len(X.columns)} features per word")
print(y.value_counts()) #Thanks to pandas.
print()
print("Splitting Data to 70-15-15: ")
print(f" > Training set:   {len(X_train)} samples")
print(f" > Validation set: {len(X_val)} samples")
print(f" > Test set:       {len(X_test)} samples")

#Note: Use print(f" "), makes it faster instead of doing the Java way.

#Task #3: Word classifier.

# OPTION A: Naive Bayes (Alexander Mejia)
#   - Faster, think of it as the "Greedy BFS" of MCO1.
#   - I think works better in this scenario, as we are only figuring out if a sentence is English or Tagalog.

# TURNS BOOLS (True and False) TO INTEGERS (1/0)
X_train_nb = X_train.astype(int)
X_val_nb = X_val.astype(int)
X_test_nb = X_test.astype(int)

modelNB = MultinomialNB(alpha=1.0)# Builds the classifier - NOTE: Add "alpha" to ensure that there is a NON-ZERO PROBABILITY (Check notes for explanation)!
modelNB.fit(X_train_nb, y_train)# Trains the model - again, ask if this is allowed. IT'S TOO EASY!

# OPTION B: Decision Tree Classifier - comment out NAIVE BAYES if attempting.

#Task #4: Validation
y_prediction_val = modelNB.predict(X_val_nb)
matrix = confusion_matrix(y_val,y_prediction_val)

#Print test:
print(classification_report(y_val, y_prediction_val, target_names=['ENG', 'FIL', 'OTH']))
print(matrix)
print("\n(Rows = Actual class, Columns = Predicted class)")
print("     ENG  FIL  OTH")
for i, label in enumerate(['ENG', 'FIL', 'OTH']):
    print(f"{label}  {matrix[i]}")

