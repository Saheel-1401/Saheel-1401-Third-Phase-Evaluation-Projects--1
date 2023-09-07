#!/usr/bin/env python
# coding: utf-8
Description: In this project/notebook, we are going to create a predictive model that predicts if an insurance claim is fraudulent or not. 
The answere between YES/NO, is a Binary Classification task. 
A comparison study has been performed to understand which ML algorithm suits best to the dataset.
# # Step 1: Load libraries and dataset

# In[1]:


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
get_ipython().run_line_magic('matplotlib', 'inline')
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


# In[3]:


data = pd.read_csv(r"C:\Users\sahee\OneDrive\Desktop\Det_Saheel\Study\Project phase evaluation\documents\FP_E_P_3\Automobile_insurance_fraud.csv")


# In[4]:


data.head(10)


# In[5]:


data.describe()


# In[6]:


# Dropping columns 
data.drop('_c39',axis=1,inplace=True)


# In[7]:


data.describe()


# In[9]:


data.dtypes


# In[11]:


data.columns


# In[12]:


data.shape


# In[14]:


data.nunique()


# In[18]:


plt.style.use('Solarize_Light2')
ax = sns.distplot(data.age, bins=np.arange(19,64,5))
ax.set_ylabel('Density')
ax.set_xlabel('Age')
plt.show()


# In[19]:


plt.style.use('fivethirtyeight')
ax = sns.countplot(x='fraud_reported', data=data, hue='fraud_reported')
ax.set_xlabel('Fraud Reported')
ax.set_ylabel('Fraud Count')
plt.show()


# In[20]:


# Count number of frauds vs non-frauds
data['fraud_reported'].value_counts() 


# In[21]:


data['incident_state'].value_counts()


# In[22]:


plt.style.use('fivethirtyeight')
fig = plt.figure(figsize=(10,6))
ax = data.groupby('incident_state').fraud_reported.count().plot.bar(ylim=0)
ax.set_ylabel('Fraud Reported')
ax.set_xlabel('Incident State')
plt.show()


# In[23]:


data['incident_state'].unique()


# In[24]:


data['collision_type'] = data['collision_type'].fillna(data['collision_type'].mode()[0])


# In[ ]:


data['property_damage'] = data['property_damage'].fillna(data['property_damage'].mode()[0])


# In[ ]:


data['police_report_available'] = data['police_report_available'].fillna(data['police_report_available'].mode()[0])


# In[26]:


data.isna().sum()


# In[27]:


# Heat Map
plt.figure(figsize = (18, 12))

corr = data.corr()

sns.heatmap(data = corr, annot = True, fmt = '.2g', linewidth = 1)
plt.show()


# In[30]:


# dropping columns which are not necessary for prediction

to_drop = ['policy_number','policy_bind_date','policy_state','insured_zip','incident_location','incident_date',
           'incident_state','incident_city','insured_hobbies','auto_make','auto_model','auto_year']

data.drop(to_drop, inplace = True, axis = 1)


# In[32]:


data.head(10)


# In[33]:


plt.figure(figsize = (18, 12))

corr = data.corr()
mask = np.triu(np.ones_like(corr, dtype = bool))

sns.heatmap(data = corr, mask = mask, annot = True, fmt = '.2g', linewidth = 1)
plt.show()

From the above plot, we can see that there is high correlation between age and months_as_customer.
We will drop the "Age" column. Also there is high correlation between total_clam_amount, injury_claim, property_claim, vehicle_claim as total claim is the sum of all others. So we will drop the total claim column.
# In[34]:


data.drop(columns = ['age', 'total_claim_amount'], inplace = True, axis = 1)


# In[35]:


data.head()


# In[36]:


data.info()


# In[37]:


# separating the feature and target columns

X = data.drop('fraud_reported', axis = 1)
y = data['fraud_reported']


# In[38]:


# extracting categorical columns
cat_data = X.select_dtypes(include = ['object'])


# In[39]:


cat_data.head()


# In[41]:


for col in cat_data.columns:
    print(f"{col}: \n{cat_data[col].unique()}\n")


# In[42]:


cat_data = pd.get_dummies(cat_data, drop_first = True)


# In[43]:


cat_data.head()


# In[44]:


num_data = X.select_dtypes(include = ['int64'])


# In[45]:


num_data.head()


# In[46]:


# combining the Numerical and Categorical dataframes to get the final dataset

X = pd.concat([num_data, cat_data], axis = 1)


# In[47]:


X.head()


# In[48]:


plt.figure(figsize = (25, 20))
plotnumber = 1

for col in X.columns:
    if plotnumber <= 24:
        ax = plt.subplot(5, 5, plotnumber)
        sns.distplot(X[col])
        plt.xlabel(col, fontsize = 15)
        
    plotnumber += 1
    
plt.tight_layout()
plt.show()


# # Outlier detection

# In[49]:


plt.figure(figsize = (20, 15))
plotnumber = 1

for col in X.columns:
    if plotnumber <= 24:
        ax = plt.subplot(5, 5, plotnumber)
        sns.boxplot(X[col])
        plt.xlabel(col, fontsize = 15)
    
    plotnumber += 1
plt.tight_layout()
plt.show()


# In[50]:


# splitting data into training set and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


# In[51]:


X_train.head()


# In[52]:


num_data = X_train[['months_as_customer', 'policy_deductable', 'umbrella_limit',
       'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
       'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'injury_claim', 'property_claim',
       'vehicle_claim']]


# In[53]:


# Scaling the numeric values in the dataset

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(num_data)


# In[54]:


scaled_num_data = pd.DataFrame(data = scaled_data, columns = num_data.columns, index = X_train.index)
scaled_num_data.head()


# In[55]:


X_train.drop(columns = scaled_num_data.columns, inplace = True)


# In[56]:


X_train = pd.concat([scaled_num_data, X_train], axis = 1)


# In[57]:


X_train.head()


# # Models
SVC
# In[58]:


from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)


# In[59]:


# accuracy_score, confusion_matrix and classification_report

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

svc_train_acc = accuracy_score(y_train, svc.predict(X_train))
svc_test_acc = accuracy_score(y_test, y_pred)

print(f"Training accuracy of Support Vector Classifier is : {svc_train_acc}")
print(f"Test accuracy of Support Vector Classifier is : {svc_test_acc}")

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

KNN
# In[64]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 30)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)


# In[65]:


# accuracy_score, confusion_matrix and classification_report

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

knn_train_acc = accuracy_score(y_train, knn.predict(X_train))
knn_test_acc = accuracy_score(y_test, y_pred)

print(f"Training accuracy of KNN is : {knn_train_acc}")
print(f"Test accuracy of KNN is : {knn_test_acc}")

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

Decision Tree Classifier
# In[66]:


from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)


# In[67]:


# accuracy_score, confusion_matrix and classification_report

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

dtc_train_acc = accuracy_score(y_train, dtc.predict(X_train))
dtc_test_acc = accuracy_score(y_test, y_pred)

print(f"Training accuracy of Decision Tree is : {dtc_train_acc}")
print(f"Test accuracy of Decision Tree is : {dtc_test_acc}")

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[68]:


# hyper parameter tuning

from sklearn.model_selection import GridSearchCV

grid_params = {
    'criterion' : ['gini', 'entropy'],
    'max_depth' : [3, 5, 7, 10],
    'min_samples_split' : range(2, 10, 1),
    'min_samples_leaf' : range(2, 10, 1)
}

grid_search = GridSearchCV(dtc, grid_params, cv = 5, n_jobs = -1, verbose = 1)
grid_search.fit(X_train, y_train)


# In[69]:


# best parameters and best score

print(grid_search.best_params_)
print(grid_search.best_score_)


# In[70]:


# best estimator 

dtc = grid_search.best_estimator_

y_pred = dtc.predict(X_test)


# In[71]:


# accuracy_score, confusion_matrix and classification_report

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

dtc_train_acc = accuracy_score(y_train, dtc.predict(X_train))
dtc_test_acc = accuracy_score(y_test, y_pred)

print(f"Training accuracy of Decision Tree is : {dtc_train_acc}")
print(f"Test accuracy of Decision Tree is : {dtc_test_acc}")

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

Random Forest Classifier
# In[72]:


from sklearn.ensemble import RandomForestClassifier

rand_clf = RandomForestClassifier(criterion= 'entropy', max_depth= 10, max_features= 'sqrt', min_samples_leaf= 1, min_samples_split= 3, n_estimators= 140)
rand_clf.fit(X_train, y_train)

y_pred = rand_clf.predict(X_test)


# In[73]:


# accuracy_score, confusion_matrix and classification_report

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

rand_clf_train_acc = accuracy_score(y_train, rand_clf.predict(X_train))
rand_clf_test_acc = accuracy_score(y_test, y_pred)

print(f"Training accuracy of Random Forest is : {rand_clf_train_acc}")
print(f"Test accuracy of Random Forest is : {rand_clf_test_acc}")

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

LGBMClassifier
# In[78]:


from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(learning_rate = 1)
lgbm.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of lgbm classifier

lgbm_acc = accuracy_score(y_test, lgbm.predict(X_test))

print(f"Training Accuracy of LGBM Classifier is {accuracy_score(y_train, lgbm.predict(X_train))}")
print(f"Test Accuracy of LGBM Classifier is {lgbm_acc} \n")

print(f"{confusion_matrix(y_test, lgbm.predict(X_test))}\n")
print(classification_report(y_test, lgbm.predict(X_test)))

Voting Classifier
# In[82]:


from sklearn.ensemble import VotingClassifier

classifiers = [('Support Vector Classifier', svc), ('KNN', knn),  ('Decision Tree', dtc), ('Random Forest', rand_clf),
               ('LGBM', lgbm)]

vc = VotingClassifier(estimators = classifiers)
vc.fit(X_train, y_train)

y_pred = vc.predict(X_test)


# In[83]:


# accuracy_score, confusion_matrix and classification_report

vc_train_acc = accuracy_score(y_train, vc.predict(X_train))
vc_test_acc = accuracy_score(y_test, y_pred)

print(f"Training accuracy of Voting Classifier is : {vc_train_acc}")
print(f"Test accuracy of Voting Classifier is : {vc_test_acc}")

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# # Model comparsion

# In[86]:


models = pd.DataFrame({
    'Model' : ['SVC', 'KNN', 'Decision Tree', 'Random Forest', 'LGBM', 'Voting Classifier'],
    'Score' : [svc_test_acc, knn_test_acc,  dtc_test_acc,rand_clf_test_acc,lgbm_acc, vc_test_acc]
})


models.sort_values(by = 'Score', ascending = False)


# In[89]:


import plotly.express as px
px.bar(data_frame = models, x = 'Score', y = 'Model', color = 'Score', template = 'plotly_dark', 
       title = 'Models Comparison')

Conclusion
Out of all the algorithms, we got best accuracy (79.6%) with SVM classifier and hyperparameter tuning.
We were able to increase our accuracy from 84% to ~90% using data cleaning, feature engineering, feature selection and hyperparameter tuning.
# In[ ]:





# In[ ]:




