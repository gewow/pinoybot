import pickle # Used to save files
from sklearn.model_selection import train_test_split # Used for 70-15-15. Splitting data.
from sklearn.naive_bayes import MultinomialNB # CHANGED FROM GAUSSIAN.
from sklearn.metrics import classification_report, confusion_matrix # IS THIS ALLOWED?! IT FEELS TOO EASY ALREADY!
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import os

# Task #1: Get data
with open('phase2_output/feature_matrix.pkl', 'rb') as f:
    X = pickle.load(f) #features 

with open('phase2_output/labels.pkl', 'rb') as f:
    y = pickle.load(f) #answers

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

# OPTION A: Multinomial Naive Bayes (Alexander Mejia)
#   - Faster, think of it as the "Greedy BFS" of MCO1.
#   - I think works better in this scenario, as we are only figuring out if a sentence is English or Filipino.

# # TURNS BOOLS (True and False) TO INTEGERS (1/0)
X_train_nb = X_train.astype(int)
X_val_nb = X_val.astype(int)
X_test_nb = X_test.astype(int)


# modelNB = MultinomialNB(alpha=1.0)# Builds the classifier - NOTE: Add "alpha" to ensure that there is a NON-ZERO PROBABILITY (Check notes for explanation)!
# modelNB.fit(X_train_nb, y_train)# Trains the model -  IT'S TOO EASY!

# OPTION B: Decision Tree Classifier - comment out NAIVE BAYES if attempting.
modelDT = DecisionTreeClassifier(random_state=50, max_depth=10) # max_depth=10 is a good start
modelDT.fit(X_train_nb, y_train)


#Task #4: Validation using the 15% validation data multinomial NB
# y_prediction_val = modelNB.predict(X_val_nb) #for multinomial NB
# matrix = confusion_matrix(y_val,y_prediction_val) 

#Print test for naive bayes:
# print(classification_report(y_val, y_prediction_val, target_names=['ENG', 'FIL', 'OTH']))
# print(matrix)
# print("\n(Rows = Actual class, Columns = Predicted class)")
# print("     ENG  FIL  OTH")
# for i, label in enumerate(['ENG', 'FIL', 'OTH']):
#     print(f"{label}  {matrix[i]}")

# OPTION B: Decision Tree Classifier
# print("--- Training Model: Decision Tree ---")
# modelDT = DecisionTreeClassifier(random_state=50, max_depth=10) # max_depth=10 is a good start
# modelDT.fit(X_train_nb, y_train)


# --- Validation: Decision Tree ---
print("\n" + "="*50)
print("VALIDATION RESULTS: DECISION TREE (OPTION B)")
print("="*50)
y_prediction_val_DT = modelDT.predict(X_val_nb) # Use the DT model to predict
matrix_DT = confusion_matrix(y_val, y_prediction_val_DT)

print(classification_report(y_val, y_prediction_val_DT, target_names=['ENG', 'FIL', 'OTH'])) # Use the DT predictions
print(matrix_DT)
print("\n(Rows = Actual class, Columns = Predicted class)")
for i, label in enumerate(['ENG', 'FIL', 'OTH']):
    print(f"{label}  {matrix_DT[i]}")


# # FINAL TESTING using the 15% test data Multionomial Naive Bayes
# print("\n" + "="*50)
# print("FINAL TEST RESULTS: Multinomial Naive Bayes")
# print("="*50)

# # Use the model to predict on the TEST set
# y_prediction_test = modelNB.predict(X_test_nb) 

# # Compare the predictions to the REAL answers
# matrix_test = confusion_matrix(y_test, y_prediction_test)

# print(classification_report(y_test, y_prediction_test, target_names=['ENG', 'FIL', 'OTH']))
# print(matrix_test)

# FINAL TESTING for Decision Tree
print("\n" + "="*50)
print("FINAL TEST RESULTS: Decision Tree")
print("="*50)

# Use the DT model to predict on the TEST set
y_prediction_test_DT = modelDT.predict(X_test_nb) 

# Compare the predictions to the REAL answers
matrix_test_DT = confusion_matrix(y_test, y_prediction_test_DT)

print(classification_report(y_test, y_prediction_test_DT, target_names=['ENG', 'FIL', 'OTH']))
print(matrix_test_DT)


#SAVING the final model

print("\n" + "="*50)
print("SAVING MODEL TO BE USED BY PINOYBOT")
print("="*50)

#create output directory if it does not exist
output_dir = 'phase3_output'
os.makedirs(output_dir, exist_ok=True)

#pkl file for the model
with open(f'{output_dir}/pinoybot_model.pkl', 'wb') as f:
    pickle.dump(modelDT, f)

#pkl file for the feature names
with open(f'{output_dir}/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

print(f"Model saved to {output_dir}/pinoybot_model.pkl")