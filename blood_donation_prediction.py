### 1.Loading the blood donations data
# Import pandas
import pandas as pd

# Read in dataset
transfusion = pd.read_csv(r'C:\Users\flash\Desktop\transfusion.data')

# Print out the first rows of our dataset
print("The first 5 rows of our dataset are : ")
print(transfusion.head())
print('\n')


### 2.Inspecting transfusion DataFrame
# Print a concise summary of transfusion DataFrame
print("Summary of transfusion dataset : ")
print(transfusion.info())
print('\n')


### 3.Creating target column
# Rename target column as 'target' for brevity 
transfusion.rename(
    columns={'whether he/she donated blood in March 2007': 'target'},
    inplace=True
)

# Print out the first 2 rows
print("Dataset after renaming column :")
print(transfusion.head(2))
print('\n')


### 4.Checking target incidence
# Print target incidence proportions, rounding output to 3 decimal places
print("Target incidence proportions, rounding output to 3 decimal places :")
print(transfusion.target.value_counts(normalize=True).round(3))
print('\n')


### 5.Splitting transfusion into train and test datasets
# Import train_test_split method
from sklearn.model_selection import train_test_split
print("Splitting transfusion into train and test datasets")

# Split transfusion DataFrame into
# X_train, X_test, y_train and y_test datasets,
# stratifying on the `target` column
X_train, X_test, y_train, y_test = train_test_split = train_test_split(
    transfusion.drop(columns='target'),
    transfusion.target,
    test_size=0.25,
    random_state=42,
    stratify=transfusion.target
)

# Print out the first 2 rows of X_train
print("First 2 rows of X_train : ")
print(X_train.head(2))
print('\n')


### 6.Selecting model using TPOT
# Import TPOTClassifier and roc_auc_score
from tpot import TPOTClassifier
from sklearn.metrics import roc_auc_score

# Instantiate TPOTClassifier
tpot = TPOTClassifier(
    generations=5,
    population_size=20,
    verbosity=2,
    scoring='roc_auc',
    random_state=42,
    disable_update_check=True,
    config_dict='TPOT light'
)
tpot.fit(X_train, y_train)

# AUC score for tpot model
print("AUC score for tpot model : ")
tpot_auc_score = roc_auc_score(y_test, tpot.predict_proba(X_test)[:, 1])
print(f'\nAUC score: {tpot_auc_score:.4f}')
print('\n')

# Print best pipeline steps
print('\nBest pipeline steps:', end='\n')
for idx, (name, transform) in enumerate(tpot.fitted_pipeline_.steps, start=1):
    # Print idx and transform
    print(f'{idx}. {transform}')


### 7.Checking the variance
print('\n')
# X_train's variance, rounding the output to 3 decimal places
print("X_train's variance, rounding the output to 3 decimal places :")
print(X_train.var().round(3))
print('\n')


### 8.Log normalization
# Import numpy
import numpy as np

# Copy X_train and X_test into X_train_normed and X_test_normed
X_train_normed, X_test_normed = X_train.copy(), X_test.copy()

# Specify which column to normalize
col_to_normalize = 'Monetary (c.c. blood)'

# Log normalization
for df_ in [X_train_normed, X_test_normed]:
    # Add log normalized column
    df_['monetary_log'] = np.log(df_[col_to_normalize])
    # Drop the original column
    df_.drop(columns=col_to_normalize, inplace=True)

# Check the variance for X_train_normed
print("Variance for X_train_normed : ")
print(X_train_normed.var().round(3))
print('\n')


### 9.Training the linear regression model
# Importing modules
from sklearn import linear_model

# Instantiate LogisticRegression
logreg = linear_model.LogisticRegression(
    solver='liblinear',
    random_state=42
)

# Train the model
logreg.fit(X_train_normed, y_train)
print("Training the linear regression model : ")

# AUC score for tpot model
logreg_auc_score = roc_auc_score(y_test, logreg.predict_proba(X_test_normed)[:, 1])
print(f'\nAUC score: {logreg_auc_score:.4f}')
print('\n')


### 10. Conclusion
# Importing itemgetter
from operator import itemgetter

# Sort models based on their AUC score from highest to lowest
print("Sort models based on their AUC score from highest to lowest : ")
print(sorted(
    [('tpot', tpot_auc_score), ('logreg', logreg_auc_score)],
    key=itemgetter(1),
    reverse=True
))

### END