#!/usr/bin/env python
# coding: utf-8

# Description: In this project/notebook, we are going to predict whether a person's income is above 50k or below 50k using various features like age, education, and occupation.

# # Step 1: Load libraries and dataset

# In[7]:


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[11]:


# Importing dataset
ds = pd.read_csv(r"C:\Users\sahee\OneDrive\Desktop\Det_Saheel\Study\Project phase evaluation\documents\FP_E_P_3\adult_census_income.csv")


# # Step 2: Descriptive analysis    

# In[12]:


# Preview dataset
ds.head()


# In[13]:


# Shape of dataset
print('Rows: {} Columns: {}'.format(ds.shape[0], ds.shape[1]))


# In[14]:


# Features data-type
ds.info()


# In[16]:


# Statistical summary
ds.describe()


# In[17]:


# Statistical summary
ds.describe().T


# In[23]:


# Check for null values
round((ds.isnull().sum()/ ds.shape[0]) * 100, 2).astype(str)+ ' %'


# In[24]:


# Check for '?' in dataset
round((dataset.isin(['?']).sum() / dataset.shape[0]) * 100, 2).astype(str) + ' %'

# Fetch Random Sample From the Dataset (50%)
# In[25]:


ds.sample(frac=0.50) #To Get Random Data At Every Execution 


# In[26]:


# Checking the counts of label categories
income = dataset['income'].value_counts(normalize=True)
round(income * 100, 2).astype('str') + ' %'

# Insight from obervation:
1) The dataset doesn't have any null values, but it contains missing values in the form of '?' which needs to be preprocessed.
2) The dataset is unbalanced, as the dependent feature 'income' contains 75.92% values have income less than 50k and 24.08% values have income more than 50k.
    
# # Step 3: Exploratory Data Analysis
3.1: Univariate Analysis
# In[33]:


# Creating a barplot for 'Income'
income = ds['income'].value_counts()

plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(7, 5))
sns.barplot(x=income.index,y=income.values, palette='bright')
plt.title('Distribution of Income', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Income', fontdict={'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=10)
plt.show()


# In[34]:


# Creating a distribution plot for 'Age'
age = ds['age'].value_counts()

plt.figure(figsize=(10, 5))
plt.style.use('fivethirtyeight')
sns.distplot(dataset['age'], bins=20)
plt.title('Distribution of Age', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Age', fontdict={'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=10)
plt.show()


# In[37]:


# Creating a barplot for 'Education'
edu = ds['education'].value_counts()

plt.style.use('seaborn')
plt.figure(figsize=(10, 5))
sns.barplot(x=edu.values, y=edu.index, palette='Paired')
plt.title('Distribution of Education', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Education', fontdict={'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=12)
plt.show()


# In[38]:


# Creating a barplot for 'Years of Education'
edu_num = ds['education.num'].value_counts()

plt.style.use('ggplot')
plt.figure(figsize=(10, 5))
sns.barplot(x=edu_num.index, y=edu_num.values, palette='colorblind')
plt.title('Distribution of Years of Education', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Years of Education', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=12)
plt.show()


# In[39]:


# Creating a pie chart for 'Marital status'
marital = ds['marital.status'].value_counts()

plt.style.use('default')
plt.figure(figsize=(10, 7))
plt.pie(marital.values, labels=marital.index, startangle=10, explode=(
    0, 0.20, 0, 0, 0, 0, 0), shadow=True, autopct='%1.1f%%')
plt.title('Marital distribution', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.legend()
plt.legend(prop={'size': 7})
plt.axis('equal')
plt.show()


# In[47]:


# Creating a donut chart for 'Age'
relation = ds['relationship'].value_counts()

plt.style.use('bmh')
plt.figure(figsize=(20, 10))
plt.pie(relation.values, labels=relation.index,
        startangle=50, autopct='%1.1f%%')
centre_circle = plt.Circle((0, 0), 0.7, fc='pink')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title('Relationship distribution', fontdict={
          'fontname': 'Monospace', 'fontsize': 30, 'fontweight': 'bold'})
plt.axis('equal')
plt.legend(prop={'size': 15})
plt.show()


# In[49]:


# Creating a barplot for 'Sex'
sex = ds['sex'].value_counts()

plt.style.use('default')
plt.figure(figsize=(7, 5))
sns.barplot(x=sex.index, y=sex.values)
plt.title('Distribution of Sex', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Sex', fontdict={'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=10)
plt.grid()
plt.show()


# In[57]:


pip install squarify


# In[58]:


# Creating a Treemap for 'Race'
import squarify
import matplotlib.pyplot as plt

race = ds['race'].value_counts()
plt.style.use('default')
plt.figure(figsize=(7, 5))
squarify.plot(sizes=race.values, label=race.index, value=race.values)
plt.title('Race distribution', fontdict={'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.show()


# In[60]:


# Creating a barplot for 'Hours per week'
hours = ds['hours.per.week'].value_counts().head(10)

plt.style.use('bmh')
plt.figure(figsize=(15, 7))
sns.barplot(x=hours.index, y=hours.values, palette='colorblind')
plt.title('Distribution of Hours of work per week', fontdict={'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Hours of work', fontdict={'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=12)
plt.show()

3.2 Bivariate Analysis
# In[65]:


# Creating a countplot of income across age
plt.style.use('default')
plt.figure(figsize=(20, 7))
sns.countplot(x=ds['age'],hue= ds['income'])
plt.title('Distribution of Income across Age', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Age', fontdict={'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=12)
plt.legend(loc=1, prop={'size': 15})
plt.show()


# In[66]:


# Creating a countplot of income across education
plt.style.use('seaborn')
plt.figure(figsize=(20, 7))
sns.countplot(x=ds['education'],
              hue=ds['income'], palette='colorblind')
plt.title('Distribution of Income across Education', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Education', fontdict={'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=12)
plt.legend(loc=1, prop={'size': 15})
plt.show()


# In[67]:


# Creating a countplot of income across years of education
plt.style.use('bmh')
plt.figure(figsize=(20, 7))
sns.countplot(x=ds['education.num'],
              hue=ds['income'])
plt.title('Income across Years of Education', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Years of Education', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=12)
plt.legend(loc=1, prop={'size': 15})
plt.savefig('bi2.png')
plt.show()


# In[68]:


# Creating a countplot of income across Marital Status
plt.style.use('seaborn')
plt.figure(figsize=(20, 7))
sns.countplot(x=ds['marital.status'], hue=ds['income'])
plt.title('Income across Marital Status', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Marital Status', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=12)
plt.legend(loc=1, prop={'size': 15})
plt.show()


# In[69]:


# Creating a countplot of income across race
plt.style.use('fivethirtyeight')
plt.figure(figsize=(20, 7))
sns.countplot(x=ds['race'], hue=ds['income'])
plt.title('Distribution of income across race', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Race', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=12)
plt.legend(loc=1, prop={'size': 15})
plt.show()


# In[70]:


# Creating a countplot of income across sex
plt.style.use('fivethirtyeight')
plt.figure(figsize=(7, 3))
sns.countplot(x=ds['sex'], hue=ds['income'])
plt.title('Distribution of income across sex', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Sex', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=12)
plt.legend(loc=1, prop={'size': 10})
plt.savefig('bi3.png')
plt.show()

3.3: Multivariate Analysis
# In[71]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[72]:


ds['income'] = le.fit_transform(dataset['income'])


# In[73]:


# Creating a pairplot of dataset
sns.pairplot(ds)
plt.savefig('multi1.png')
plt.show()


# In[74]:


corr = ds.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(7, 5))
    ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True,
                     annot=True, cmap='RdYlGn')
plt.savefig('multi2.png')
plt.show()

In this dataset, the most number of people are young, white, male, high school graduates with 9 to 10 years of education and work 40 hours per week.

From the correlation heatmap, we can see that the dependent feature 'income' is highly correlated with age, numbers of years of education, capital gain and number of hours per week.
# # Step 4: Data Preprocessing

# In[75]:


ds = ds.replace('?', np.nan)


# In[76]:


# Checking null values
round((ds.isnull().sum() / ds.shape[0]) * 100, 2).astype(str) + ' %'


# In[77]:


columns_with_nan = ['workclass', 'occupation', 'native.country']


# In[78]:


for col in columns_with_nan:
    ds[col].fillna(ds[col].mode()[0], inplace=True)


# In[79]:


from sklearn.preprocessing import LabelEncoder


# In[92]:


for col in ds.columns:
    if ds[col].dtypes == 'object':
        encoder = LabelEncoder()
        ds[col] = encoder.fit_transform(ds[col])


# In[93]:


X = ds.drop('income', axis=1)
Y = ds['income']


# In[94]:


from sklearn.ensemble import ExtraTreesClassifier
selector = ExtraTreesClassifier(random_state=42)


# In[95]:


selector.fit(X, Y)


# In[88]:


ExtraTreesClassifier(random_state=42)


# In[96]:


feature_imp = selector.feature_importances_


# In[97]:


for index, val in enumerate(feature_imp):
    print(index, round((val * 100), 2))


# In[98]:


X.info()


# In[99]:


X = X.drop(['workclass', 'education', 'race', 'sex',
            'capital.loss', 'native.country'], axis=1)


# In[100]:


from sklearn.preprocessing import StandardScaler


# In[101]:


for col in X.columns:
    scaler = StandardScaler()
    X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))


# In[102]:


round(Y.value_counts(normalize=True) * 100, 2).astype('str') + ' %'


# In[103]:


from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)


# In[104]:


ros.fit(X, Y)


# In[105]:


X_resampled, Y_resampled = ros.fit_resample(X, Y)


# In[106]:


round(Y_resampled.value_counts(normalize=True) * 100, 2).astype('str') + ' %'

 Creating a train test split
# In[107]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X_resampled, Y_resampled, test_size=0.2, random_state=42)


# In[108]:


print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)


# # Step 5:Data Modelling
# Logistic regression
# In[109]:


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=42)


# In[110]:


log_reg.fit(X_train, Y_train)


# In[115]:


Y_pred_log_reg = log_reg.predict(X_test)

# KNN Classifier
# In[112]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()


# In[113]:


knn.fit(X_train, Y_train)


# In[114]:


Y_pred_knn = knn.predict(X_test)

# Naive Bayes Classifier
# In[116]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()


# In[117]:


nb.fit(X_train, Y_train)


# In[118]:


GaussianNB()


# In[119]:


Y_pred_nb = nb.predict(X_test)

# Decision Tree Classifier
# In[120]:


from sklearn.tree import DecisionTreeClassifier
dec_tree = DecisionTreeClassifier(random_state=42)


# In[121]:


dec_tree.fit(X_train, Y_train)


# In[122]:


Y_pred_dec_tree = dec_tree.predict(X_test)

# Random Forest Classifier
# In[123]:


from sklearn.ensemble import RandomForestClassifier
ran_for = RandomForestClassifier(random_state=42)


# In[124]:


ran_for.fit(X_train, Y_train)


# In[125]:


Y_pred_ran_for = ran_for.predict(X_test)

# XGB Classifier
# In[127]:


from xgboost import XGBClassifier
xgb = XGBClassifier()


# In[128]:


xgb.fit(X_train, Y_train)


# In[129]:


Y_pred_xgb = xgb.predict(X_test)


# # Step 6: Model Evaluation

# In[132]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


# In[133]:


print('Logistic Regression:')
print('Accuracy score:', round(accuracy_score(Y_test, Y_pred_log_reg) * 100, 2))
print('F1 score:', round(f1_score(Y_test, Y_pred_log_reg) * 100, 2))


# In[134]:


print('KNN Classifier:')
print('Accuracy score:', round(accuracy_score(Y_test, Y_pred_knn) * 100, 2))
print('F1 score:', round(f1_score(Y_test, Y_pred_knn) * 100, 2))


# In[135]:


print('Naive Bayes Classifier:')
print('Accuracy score:', round(accuracy_score(Y_test, Y_pred_nb) * 100, 2))
print('F1 score:', round(f1_score(Y_test, Y_pred_nb) * 100, 2))


# In[136]:


print('Decision Tree Classifier:')
print('Accuracy score:', round(accuracy_score(Y_test, Y_pred_dec_tree) * 100, 2))
print('F1 score:', round(f1_score(Y_test, Y_pred_dec_tree) * 100, 2))


# In[137]:


print('Random Forest Classifier:')
print('Accuracy score:', round(accuracy_score(Y_test, Y_pred_ran_for) * 100, 2))
print('F1 score:', round(f1_score(Y_test, Y_pred_ran_for) * 100, 2))


# In[138]:


print('XGB Classifier:')
print('Accuracy score:', round(accuracy_score(Y_test, Y_pred_xgb) * 100, 2))
print('F1 score:', round(f1_score(Y_test, Y_pred_xgb) * 100, 2))


# # Step 7: Hyperparameter Tuning

# In[148]:


from sklearn.model_selection import RandomizedSearchCV


# In[149]:


n_estimators = [int(x) for x in np.linspace(start=40, stop=150, num=15)]
max_depth = [int(x) for x in np.linspace(40, 150, num=15)]


# In[150]:


param_dist = {
    'n_estimators': n_estimators,
    'max_depth': max_depth,
}


# In[142]:


rf_tuned = RandomForestClassifier(random_state=42)


# In[151]:


rf_cv = RandomizedSearchCV(
    estimator=rf_tuned, param_distributions=param_dist, cv=5, random_state=42)


# In[144]:


rf_cv.fit(X_train, Y_train)


# In[145]:


rf_cv.best_score_


# In[153]:


rf_best = RandomForestClassifier(
    max_depth=102, n_estimators=40, random_state=42)


# In[154]:


rf_best.fit(X_train, Y_train)


# In[155]:


Y_pred_rf_best = rf_best.predict(X_test)


# In[156]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred_rf_best)


# In[157]:


plt.style.use('default')
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
plt.savefig('heatmap.png')
plt.show()


# In[158]:


from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred_rf_best))

# Conclusion:
1) A hyperparameter tuned random forest classifier gives the highest accuracy score of 92.77 and f1 score of 93.08.
# In[ ]:




