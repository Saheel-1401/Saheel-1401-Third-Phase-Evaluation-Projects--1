#!/usr/bin/env python
# coding: utf-8
Rainfall Prediction is one of the difficult and uncertain tasks that have a significant impact on human society. 
The following project offers a series of experiments that employ well-known machine learning methods to build models that can forecast whether it will rain or not tomorrow based on the weather data for that day in significant Australian cities.
# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
wdata = pd.read_csv(r"C:\Users\sahee\OneDrive\Desktop\Det_Saheel\Study\Project phase evaluation\documents\FP_E_P_3\weatherAUS.csv")


# In[2]:


wdata.head()


# In[3]:


wdata['Date'] = pd.to_datetime(wdata['Date'])
wdata['year'] = wdata['Date'].dt.year
wdata['month'] = wdata['Date'].dt.month
wdata['day'] = wdata['Date'].dt.day
wdata.drop(['Date'], axis = 1,inplace=True) 


# In[4]:


wdata.shape


# In[5]:


wdata.info()


# In[6]:


wdata['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)
wdata['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)
wdata.head()


# In[7]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize = (20,5))
ax=wdata.RainTomorrow.value_counts(normalize = True).plot(kind='bar', color= ['skyblue','navy'], alpha = 0.9, rot=0)
plt.title('RainTomorrow Indicator No(0) and Yes(1) in the Imbalanced Dataset')
for p in ax.patches:
    ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.01 , p.get_height() * 1.01))
plt.show()


# In[8]:


from sklearn.utils import resample

no = wdata[wdata.RainTomorrow == 0]
yes = wdata[wdata.RainTomorrow == 1]
yes_oversampled = resample(yes, replace=True, n_samples=len(no), random_state=42)
oversampled = pd.concat([no, yes_oversampled])

fig = plt.figure(figsize = (20,5))
ax=oversampled.RainTomorrow.value_counts(normalize = True).plot(kind='bar', color= ['skyblue','navy'], alpha = 0.9, rot=0)
plt.title('RainTomorrow Indicator No(0) and Yes(1) after Oversampling (Balanced Dataset)')
for p in ax.patches:
    ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.01 , p.get_height() * 1.01))
plt.show()


# # Missing Data Pattern in Training Data

# In[9]:


import seaborn as sns
plt.figure(figsize = (20,5))
sns.heatmap(oversampled.isnull(), cbar=False, cmap='PuBu')
plt.show()


# In[10]:


total = oversampled.isnull().sum().sort_values(ascending=False)
percent = (oversampled.isnull().sum()/oversampled.isnull().count()).sort_values(ascending=False)
missing = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing.head()


# In[11]:


oversampled.select_dtypes(include=['object']).columns


# In[12]:


# Impute categorical var with Mode
oversampled['Location'] = oversampled['Location'].fillna(oversampled['Location'].mode()[0])
oversampled['WindGustDir'] = oversampled['WindGustDir'].fillna(oversampled['WindGustDir'].mode()[0])
oversampled['WindDir9am'] = oversampled['WindDir9am'].fillna(oversampled['WindDir9am'].mode()[0])
oversampled['WindDir3pm'] = oversampled['WindDir3pm'].fillna(oversampled['WindDir3pm'].mode()[0])


# In[13]:


# Convert categorical features to continuous features with Label Encoding
from sklearn.preprocessing import LabelEncoder
lencoders = {}
for col in oversampled.select_dtypes(include=['object']).columns:
    lencoders[col] = LabelEncoder()
    oversampled[col] = lencoders[col].fit_transform(oversampled[col])


# In[14]:


import warnings
warnings.filterwarnings("ignore")


# In[15]:


# Multiple Imputation by Chained Equations
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
MiImputed = oversampled.copy(deep=True) 
mi_imputer = IterativeImputer()
MiImputed.iloc[:, :] = mi_imputer.fit_transform(oversampled)


# In[16]:


MiImputed.head()


# In[17]:


MiImputed.isna()


# In[18]:


# Detecting outliers with IQR
Q1 = MiImputed.quantile(0.25)
Q3 = MiImputed.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[19]:


# Removing outliers from dataset
MiImputed = MiImputed[~((MiImputed < (Q1 - 1.5 * IQR)) |(MiImputed > (Q3 + 1.5 * IQR))).any(axis=1)]
MiImputed.shape

We observe that the original dataset was having the shape (12390, 25). 
After running outlier-removal code snippet, the dataset is now having the shape (9733, 25). 
So, the dataset is now free of 2657 outliers. 
We will now check for multi-collinearity i.e. whether any feature is highly correlated with another.
# In[20]:


# Correlation Heatmap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
corr = MiImputed.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(20, 20))
cmap = sns.diverging_palette(250, 25, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=None, center=0,square=True, annot=True, linewidths=.5, cbar_kws={"shrink": .9})


# In[21]:


sns.pairplot( data=MiImputed, vars=('MaxTemp','MinTemp','Pressure9am',
                                    'Pressure3pm', 'Temp9am', 'Temp3pm', 'Evaporation'), hue='RainTomorrow' )


# # Feature Selection

# In[22]:


# Standardizing data
from sklearn import preprocessing
r_scaler = preprocessing.MinMaxScaler()
r_scaler.fit(MiImputed)
modified_data = pd.DataFrame(r_scaler.transform(MiImputed), index=MiImputed.index, columns=MiImputed.columns)
modified_data.head()


# In[23]:


# Feature Importance using Filter Method (Chi-Square)
from sklearn.feature_selection import SelectKBest, chi2
X = modified_data.loc[:,modified_data.columns!='RainTomorrow']
y = modified_data[['RainTomorrow']]
selector = SelectKBest(chi2, k=10)
selector.fit(X, y)
X_new = selector.transform(X)
print(X.columns[selector.get_support(indices=True)])


# In[24]:


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as rf

X = MiImputed.drop('RainTomorrow', axis=1)
y = MiImputed['RainTomorrow']
selector = SelectFromModel(rf(n_estimators=100, random_state=0))
selector.fit(X, y)
support = selector.get_support()
features = X.loc[:,support].columns.tolist()
print(features)
print(rf(n_estimators=100, random_state=0).fit(X,y).feature_importances_)


# In[25]:


import warnings
warnings.filterwarnings("ignore")


# In[26]:


pip install eli5


# In[27]:


pip install scikit-learn


# In[28]:


import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(rf(n_estimators=100, random_state=0).fit(X,y),random_state=1).fit(X,y)
eli5.show_weights(perm, feature_names = X.columns.tolist())


# # Training with Different Models

# In[ ]:


features = MiImputed[['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 
                       'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 
                       'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 
                       'RainToday']]
target = MiImputed['RainTomorrow']


# In[ ]:


# Split into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=12345)

# Normalize Features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# In[ ]:


def plot_roc_cur(fper, tper):  
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


# In[ ]:


pip install --upgrade scikit-learn


# In[ ]:


conda update -c conda-forge scikit-learn


# In[51]:


import time
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, roc_curve, classification_report
def run_model(model, X_train, y_train, X_test, y_test, verbose=True):
    t0=time.time()
    if verbose == False:
        model.fit(X_train,y_train, verbose=0)
    else:
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred) 
        coh_kap = cohen_kappa_score(y_test, y_pred)
        time_taken = time.time()-t0
    print("Accuracy = {}".format(accuracy))
    print("ROC Area under Curve = {}".format(roc_auc))
    print("Cohen's Kappa = {}".format(coh_kap))
    print("Time taken = {}".format(time_taken))
    print(classification_report(y_test,y_pred,digits=5))
    
    probs = model.predict_proba(X_test)  
    probs = probs[:, 1]  
    fper, tper, thresholds = roc_curve(y_test, probs) 
    plot_roc_cur(fper, tper)
    
    return model, accuracy, roc_auc, coh_kap, time_taken


# # Model-1:- Logistic Regression penalized by Lasso

# In[53]:


from sklearn.linear_model import LogisticRegression as lr

params_lr = {'penalty': 'l1', 'solver':'liblinear'}

model_lr = lr(**params_lr)
run_model(model_lr, X_train, y_train, X_test, y_test)


# # Model-2:- Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

params_dt = {'max_depth': 16,
             'max_features': "sqrt"}

model_dt = DecisionTreeClassifier(**params_dt)
model_dt, accuracy_dt, roc_auc_dt, coh_kap_dt, tt_dt = run_model(model_dt, X_train, y_train, X_test, y_test)


# # Model-3:- Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

params_rf = {'max_depth': 16,
             'min_samples_leaf': 1,
             'min_samples_split': 2,
             'n_estimators': 100,
             'random_state': 12345}

model_rf = RandomForestClassifier(**params_rf)
model_rf, accuracy_rf, roc_auc_rf, coh_kap_rf, tt_rf = run_model(model_rf, X_train, y_train, X_test, y_test)


# # Model-4:- Light GBM

# In[ ]:


import lightgbm as lgb
params_lgb ={'colsample_bytree': 0.95, 
         'max_depth': 16, 
         'min_split_gain': 0.1, 
         'n_estimators': 200, 
         'num_leaves': 50, 
         'reg_alpha': 1.2, 
         'reg_lambda': 1.2, 
         'subsample': 0.95, 
         'subsample_freq': 20}

model_lgb = lgb.LGBMClassifier(**params_lgb)
model_lgb, accuracy_lgb, roc_auc_lgb, coh_kap_lgb, tt_lgb = run_model(model_lgb, X_train, y_train, X_test, y_test)


# # Model-5:- XGBoost

# In[ ]:


import xgboost as xgb
params_xgb ={'n_estimators': 500,
            'max_depth': 16}

model_xgb = xgb.XGBClassifier(**params_xgb)
model_xgb, accuracy_xgb, roc_auc_xgb, coh_kap_xgb, tt_xgb = run_model(model_xgb, X_train, y_train, X_test, y_test)


# # Plotting Decision Region for all Models

# In[ ]:


pip install mlxtend


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
from sklearn.linear_model import LogisticRegression as lr
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.ensemble import RandomForestClassifier as rf
import lightgbm as lgb
import xgboost as xgb
from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.plotting import plot_decision_regions

value = 1.80
width = 0.90

clf1 = lr(random_state=12345)
clf2 = dt(random_state=12345) 
clf4 = rf(random_state=12345)
clf5 = lgb.LGBMClassifier(random_state=12345, verbose = 0)
clf7 = xgb.XGBClassifier(random_state=12345)
eclf = EnsembleVoteClassifier(clfs=[clf4, clf5,clf7], weights=[1, 1, 1], voting='soft')

X_list = MiImputed[["Sunshine", "Humidity9am", "Cloud3pm"]] #took only really important features
X = np.asarray(X_list, dtype=np.float32)
y_list = MiImputed["RainTomorrow"]
y = np.asarray(y_list, dtype=np.int32)

# Plotting Decision Regions
gs = gridspec.GridSpec(3,3)
fig = plt.figure(figsize=(18, 14))

labels = ['Logistic Regression',
          'Decision Tree',
          'Random Forest',
          'LightGBM',
          'XGBoost',
          'Ensemble']

for clf, lab, grd in zip([clf1, clf2, clf4, clf5,clf7, eclf],
                         labels,
                         itertools.product([0, 1, 2],
                         repeat=2)):
    clf.fit(X, y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=y, clf=clf, 
                                filler_feature_values={2: value}, 
                                filler_feature_ranges={2: width}, 
                                legend=2)
    plt.title(lab)

plt.show()


# In[ ]:


import sklearn
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
# Assuming tt represents a placeholder time value in seconds
tt = 0.0  # You can replace 0.0 with the actual time taken in seconds

import time  # Import the time module
start_time = time.time()  # Record the start time
end_time = time.time()  # Record the end time
tt = end_time - start_time  # Calculate the time taken in seconds


accuracy_score=[accuracy_lr,accuracy_dt, accuracy_rf, accuracy_lgb,accuracy_xgb]
roc_auc_score=[roc_auc_lr, roc_auc_dt, roc_auc_rf, roc_auc_lgb,roc_auc_xgb]
cohen_kappa_score=[cohen_kappa_lr, cohen_kappa_dt, cohen_kappa_rf, cohen_kappa_lgb,cohen_kappa_xgb]
Time_taken=[lr,dt,rf,lgb,xgb]

model_data ={'Model':['Logistic Regression','Decision Tree','Random Forest','LightGBM','XGBoost'],
             'Accuracy': accuracy_score,
              'ROC_AUC': roc_auc_score,
              'Cohen_Kappa': cohen_kappa_score,
              'Time_taken': tt}
import pandas as pd
data = pd.DataFrame(model_data)

fig, ax1 = plt.subplots(figsize=(12,10))
ax1.set_title('Model Comparison: Accuracy and Time taken for execution', fontsize=13)
color = 'tab:green'
ax1.set_xlabel('Model', fontsize=13)
ax1.set_ylabel('Time_taken', fontsize=13, color=color)
ax2 = sns.barplot(x='Model', y='Time_taken', data = model_data, palette='summer')
ax1.tick_params(axis='y')
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Accuracy', fontsize=13, color=color)
ax2 = sns.lineplot(x='Model', y='Accuracy', data = model_data, sort=False, color=color)
ax2.tick_params(axis='y', color=color)


# In[5]:


import pandas as pd
from sklearn.metrics import accuracy_score  # Import accuracy_score function
from sklearn.metrics import roc_auc_score, cohen_kappa_score
import matplotlib.pyplot as plt

model_data ={'Model':['Logistic Regression','Decision Tree','Random Forest','LightGBM','XGBoost'],
             'Accuracy': accuracy_score,
              'ROC_AUC': roc_auc_score,
              'Cohen_Kappa': cohen_kappa_score,
              'Time_taken': tt}  # Now 'tt' is defined and assigned a value

data_1 = pd.DataFrame(model_data)
fig, ax3 = plt.subplots(figsize=(12,10))
ax3.set_title('Model Comparison: Area under ROC and Cohens Kappa', fontsize=13)
color = 'tab:blue'
ax3.set_xlabel('Model', fontsize=13)
ax3.set_ylabel('ROC_AUC', fontsize=13, color=color)
ax4 = sns.barplot(x ='Model', y='ROC_AUC', data_1 = model_data, palette='winter')
ax3.tick_params(axis='y')
ax4 = ax3.twinx()
color = 'tab:red'
ax4.set_ylabel('Cohen_Kappa', fontsize=13, color=color)
ax4 = sns.lineplot(x='Model', y='Cohen_Kappa',data_1 = model_data, sort=False, color=color)
ax4.tick_params(axis='y', color=color)
plt.show()


# In[ ]:




Conclusion
We can observe that XGBoost and Random Forest have performed better compared to other models. 
However, if speed is an important thing to consider, we can stick to Random Forest instead of XGBoost.
# In[ ]:




