# Author: Alperen Kantarcı
# Machine Learning Term Project

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import * 
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import seaborn as sns

pd.set_option('display.max_columns', 100)

# Required paths
train_values_p = "train_values.csv"
train_labels_p = "train_labels.csv"
test_values_p ="test_values.csv"
# State
random_token = 1771

train_vals = pd.read_csv(train_values_p)
train_labels = pd.read_csv(train_labels_p)
test_vals = pd.read_csv(test_values_p)

merged_train_df = pd.merge(train_vals, train_labels, on='building_id', how='outer')

plt.hist(merged_train_df['age'],bins=100)
plt.show()
unique = merged_train_df['age'].unique()
unique.sort()
# Print ages of the buildings to see overview
print(unique)

plt.figure(figsize=(20,15))  
sns.heatmap(merged_train_df.corr(), annot=True, fmt=".2f")
plt.show()

train_vals.head()

dummies = pd.get_dummies(train_vals["land_surface_condition"],prefix="land_surface_condition")
train_vals = pd.concat([train_vals,dummies],axis='columns')
train_vals = train_vals.drop(columns="land_surface_condition")

dummies = pd.get_dummies(train_vals["foundation_type"],prefix="foundation_type")
train_vals = pd.concat([train_vals,dummies],axis='columns')
train_vals = train_vals.drop(columns="foundation_type")

dummies = pd.get_dummies(train_vals["roof_type"],prefix="roof_type")
train_vals = pd.concat([train_vals,dummies],axis='columns')
train_vals = train_vals.drop(columns="roof_type")

dummies = pd.get_dummies(train_vals["ground_floor_type"],prefix="ground_floor_type")
train_vals = pd.concat([train_vals,dummies],axis='columns')
train_vals = train_vals.drop(columns="ground_floor_type")

dummies = pd.get_dummies(train_vals["other_floor_type"],prefix="other_floor_type")
train_vals = pd.concat([train_vals,dummies],axis='columns')
train_vals = train_vals.drop(columns="other_floor_type")


dummies = pd.get_dummies(train_vals["position"],prefix="position")
train_vals = pd.concat([train_vals,dummies],axis='columns')
train_vals = train_vals.drop(columns="position")

dummies = pd.get_dummies(train_vals["plan_configuration"],prefix="plan_configuration")
train_vals = pd.concat([train_vals,dummies],axis='columns')
train_vals = train_vals.drop(columns="plan_configuration")


dummies = pd.get_dummies(train_vals["legal_ownership_status"],prefix="legal_ownership_status")
train_vals = pd.concat([train_vals,dummies],axis='columns')
train_vals = train_vals.drop(columns="legal_ownership_status")

# New data features
train_vals.head()
train_labels

X_train, X_val, y_train, y_val = train_test_split(train_vals, train_labels, test_size = 0.25, random_state = random_token )

pipe = make_pipeline(StandardScaler(), 
                     RandomForestClassifier(random_state=random_token))
param_grid = {'randomforestclassifier__n_estimators': [1 ,20 , 50],
              'randomforestclassifier__min_samples_leaf': [1, 5, 20]}
gs = GridSearchCV(pipe, param_grid, cv=5)
print("Training takes time, please wait.")
y_train = y_train.iloc[:,1].values
gs.fit(X_train, y_train)

y_val = y_val.iloc[:,1].values
predictions = gs.predict(X_val)
true = (y_val == predictions).sum()
false = (y_val != predictions).sum()
print("Validation Set True:{} False:{}, Accuracy:{}".format(true,false,100*true/(true+false)))

# Make deeper investigation on the validation set
from sklearn.metrics import classification_report
print(classification_report(y_val, predictions))

# Test set preprocessing
dummies = pd.get_dummies(test_vals["land_surface_condition"],prefix="land_surface_condition")
test_vals = pd.concat([test_vals,dummies],axis='columns')
test_vals = test_vals.drop(columns="land_surface_condition")

dummies = pd.get_dummies(test_vals["foundation_type"],prefix="foundation_type")
test_vals = pd.concat([test_vals,dummies],axis='columns')
test_vals = test_vals.drop(columns="foundation_type")

dummies = pd.get_dummies(test_vals["roof_type"],prefix="roof_type")
test_vals = pd.concat([test_vals,dummies],axis='columns')
test_vals = test_vals.drop(columns="roof_type")

dummies = pd.get_dummies(test_vals["ground_floor_type"],prefix="ground_floor_type")
test_vals = pd.concat([test_vals,dummies],axis='columns')
test_vals = test_vals.drop(columns="ground_floor_type")

dummies = pd.get_dummies(test_vals["other_floor_type"],prefix="other_floor_type")
test_vals = pd.concat([test_vals,dummies],axis='columns')
test_vals = test_vals.drop(columns="other_floor_type")


dummies = pd.get_dummies(test_vals["position"],prefix="position")
test_vals = pd.concat([test_vals,dummies],axis='columns')
test_vals = test_vals.drop(columns="position")

dummies = pd.get_dummies(test_vals["plan_configuration"],prefix="plan_configuration")
test_vals = pd.concat([test_vals,dummies],axis='columns')
test_vals = test_vals.drop(columns="plan_configuration")


dummies = pd.get_dummies(test_vals["legal_ownership_status"],prefix="legal_ownership_status")
test_vals = pd.concat([test_vals,dummies],axis='columns')
test_vals = test_vals.drop(columns="legal_ownership_status")

predictions = gs.predict(test_vals)

df = pd.DataFrame()
df['building_id'] = np.array(test_vals["building_id"]).T
df['damage_grade'] = np.array(predictions,dtype=np.int64).T
df

df.to_csv('solution.csv', index = False)
