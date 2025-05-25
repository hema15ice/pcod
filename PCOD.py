#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv("PCOS_data.csv")


# In[3]:


data.head()


# In[4]:


data.columns


# In[5]:


data =data.drop('Unnamed: 44', axis=1)


# In[6]:


data.head()


# In[7]:


data = data.rename(columns = {"PCOS (Y/N)":"Target"})


# In[8]:


data.head()


# In[9]:


data = data.drop(["Sl. No","Patient File No."],axis = 1)


# In[10]:


data.info(verbose = True, null_counts = False)


# In[11]:


data["AMH(ng/mL)"].head() 


# In[12]:


data["II    beta-HCG(mIU/mL)"].head()


# In[13]:


data["AMH(ng/mL)"] = pd.to_numeric(data["AMH(ng/mL)"], errors='coerce')

data["II    beta-HCG(mIU/mL)"] = pd.to_numeric(data["II    beta-HCG(mIU/mL)"], errors='coerce')


# In[14]:


# Dealing with missing values. 
# Filling NA values with the median of that feature.

data['Marraige Status (Yrs)'].fillna(data['Marraige Status (Yrs)'].median(),inplace=True)

data['II    beta-HCG(mIU/mL)'].fillna(data['II    beta-HCG(mIU/mL)'].median(),inplace=True)

data['AMH(ng/mL)'].fillna(data['AMH(ng/mL)'].median(),inplace=True)

data['Fast food (Y/N)'].fillna(data['Fast food (Y/N)'].median(),inplace=True)


# In[15]:


# Clearing up the extra space in the column names.

data.columns = [col.strip() for col in data.columns]


# In[16]:


colors = ['#670067','#008080']


# In[17]:


def bar_plot(variable):
    """
     input: variable example : Target
     output: bar plot & value count
     
    """
    # Get feature
    var = data[variable]
    # Count number of categorical variable(value/sample)
    varValue = var.value_counts()
    # Visualize
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index,varValue,color=colors)
    plt.xticks(varValue.index,varValue.index.values)
    plt.ylabel("Count")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable,varValue))


# In[18]:


category = ["Target", "Pregnant(Y/N)", "Weight gain(Y/N)", "hair growth(Y/N)", "Skin darkening (Y/N)", "Hair loss(Y/N)", "Pimples(Y/N)", "Fast food (Y/N)", "Reg.Exercise(Y/N)", "Blood Group"]
for c in category:
    bar_plot(c)


# In[19]:


def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(data[variable], bins = 50,color=colors[1])
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()


# In[20]:


numericVar = ["Age (yrs)", "Weight (Kg)","Marraige Status (Yrs)"]
for n in numericVar:
    plot_hist(n)


# In[21]:


#EDA
# Having a look at some basic statistical details.

data.describe()


# In[22]:


# Examaning a correlation matrix of all the features.

corrmat = data.corr()
plt.subplots(figsize=(18,18))
sns.heatmap(corrmat,cmap="Set3", square=True);


# In[23]:


# How all the features correlate with the PCOS. 

corrmat['Target'].sort_values(ascending=False)


# In[24]:


# Having a look at features bearing significant correlation.

plt.figure(figsize=(12,12))
k = 12 #number of variables with positive for heatmap
l = 3 #number of variables with negative for heatmap
cols_p = corrmat.nlargest(k,'Target')['Target'].index 
cols_n = corrmat.nsmallest(l, 'Target')['Target'].index
cols = cols_p.append(cols_n) 

cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True,cmap="Set3", annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                 yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[25]:


# Length of menstrual phase in PCOS vs normal 

fig=sns.lmplot(data=data,x="Age (yrs)",y="Cycle length(days)", hue="Target",palette=colors)
plt.show(fig)


# In[26]:


# Pattern of weight gain (BMI) over years in PCOS and Normal. 

fig= sns.lmplot(data =data,x="Age (yrs)",y="BMI", hue="Target", palette= colors )
plt.show(fig)


# In[27]:


# Cycle IR wrt age 

sns.lmplot(data =data,x="Age (yrs)",y="Cycle(R/I)", hue="Target",palette=colors)
plt.show()


# In[28]:


# Distribution of follicles in both ovaries.

sns.lmplot(data =data,x='Follicle No. (R)',y='Follicle No. (L)', hue="Target",palette=colors)
plt.show()


# In[29]:


# Exploring the above observation with the help of Boxplot

color = ["teal", "plum"]
features = ["Follicle No. (L)","Follicle No. (R)"]
for i in features:
    sns.swarmplot(x=data["Target"], y=data[i], color="black", alpha=0.5 )
    sns.boxenplot(x=data["Target"], y=data[i], palette=color)
    plt.show()


# In[30]:


features = ["Age (yrs)","Weight (Kg)", "BMI", "Hb(g/dl)", "Cycle length(days)","Endometrium (mm)" ]
for i in features:
    sns.swarmplot(x=data["Target"], y=data[i], color="black", alpha=0.5 )
    sns.boxenplot(x=data["Target"], y=data[i], palette=color)
    plt.show()


# In[31]:


x= data.drop(labels = ["Target"],axis = 1)
y=data.Target


# In[32]:


from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)


# In[33]:


print("x_train",len(x_train))
print("x_test",len(x_test))
print("y_train",len(y_train))
print("y_test",len(y_test))


# In[34]:


from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(x_train,y_train)


# In[35]:


acc_log_train = round(lg.score(x_train, y_train)*100,2) 
acc_log_test = round(lg.score(x_test,y_test)*100,2)


# In[36]:


print("Training Accuracy: % {}".format(acc_log_train))
print("Testing Accuracy: % {}".format(acc_log_test))


# In[37]:


pip install catboost


# In[38]:


pip install xgboost


# In[39]:


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


# In[40]:


random_state = 42
classifier = [DecisionTreeClassifier(random_state = random_state),
             SVC(random_state = random_state),
             RandomForestClassifier(random_state = random_state),
             LogisticRegression(random_state = random_state),
             KNeighborsClassifier()]


# In[41]:


# Decision Tree
dt= {"min_samples_split" : range(10,500,20),
                "max_depth": range(1,20,2)}


# In[42]:


# SVM
svc = {"kernel" : ["rbf"],
                 "gamma": [0.001, 0.01, 0.1, 1],
                 "C": [1,10,50,100,200,300,1000]}


# In[43]:


# Random Forest
rf= {"max_features": ['auto', 'sqrt', 'log2'],
                "n_estimators":[300,500],
                "criterion":["gini"],
                'max_depth' : [4,5,6,7,8,9,10,12],}


# In[44]:


# Logistic Regression
logreg= {"C":np.logspace(-3,3,7),
                    "penalty": ["l1","l2"]}


# In[45]:


# KNN
knn= {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),
                 "weights": ["uniform","distance"],
                 "metric":["euclidean","manhattan"]}


# In[46]:


classifier_param = [dt,
                   svc,
                   rf,
                   logreg,
                   knn]


# In[47]:


from sklearn.model_selection import StratifiedKFold, GridSearchCV
cv_result = []
best_estimators = []
for i in range(len(classifier)):
    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10),
                       scoring = "accuracy", n_jobs = -1,verbose = 1)
    clf.fit(x_train,y_train)
    cv_result.append(round(clf.best_score_*100,2))
    best_estimators.append(clf.best_estimator_)
    print(cv_result[i])


# In[48]:


best_estimators


# In[49]:


dt = best_estimators[0]
svm = best_estimators[1]
rf = best_estimators[2]
logreg = best_estimators[3]
knn = best_estimators[4]


# In[50]:


# XGBRF Classifier
import xgboost
xgb_clf = xgboost.XGBRFClassifier(max_depth=3, random_state=random_state)
xgb_clf.fit(x_train,y_train)
acc_xgb_clf_train = round(xgb_clf.score(x_train, y_train)*100,2) 
acc_xgb_clf_test = round(xgb_clf.score(x_test,y_test)*100,2)
cv_result.append(acc_xgb_clf_train)
print("Training Accuracy: % {}".format(acc_xgb_clf_train))
print("Testing Accuracy: % {}".format(acc_xgb_clf_test))


# In[51]:


pip install catboost


# In[52]:


# CatBoost Classifier
from catboost import CatBoostClassifier

cat_clf = CatBoostClassifier()
cat_clf.fit(x_train,y_train)
acc_cat_clf_train = round(cat_clf.score(x_train, y_train)*100,2) 
acc_cat_clf_test = round(cat_clf.score(x_test,y_test)*100,2)
cv_result.append(acc_cat_clf_train)
print("Training Accuracy: % {}".format(acc_cat_clf_train))
print("Testing Accuracy: % {}".format(acc_cat_clf_test))


# In[53]:


model_list = ['Decision Tree','SVC','RandomForest','Logistic Regression','KNearestNeighbours','XGBRF','CatBoostClassifier']


# In[54]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')
fg = sns.catplot(x=model_list, y=cv_result, height=7, aspect=2, kind='bar', data=data)
plt.title('Accuracy of different Classifier Models')
plt.xlabel('Classifier Models')
plt.show()


# In[55]:


# Plotly Bar Chart:

import plotly.graph_objects as go
trace1 = go.Bar(
                x = model_list,
                y = cv_result,
                marker = dict(color = 'rgb(32, 55, 110)',
                              line=dict(color='rgb(0,0,0)',width=1.5)))
layout = go.Layout(title = 'Accuracy of different Classifier Models' , xaxis = dict(title = 'Classifier Models'), yaxis = dict(title = '% of Accuracy'))
fig = go.Figure(data = [trace1], layout = layout)
fig.show()


# In[56]:


model = [dt,svm,rf,logreg,knn,xgb_clf,cat_clf]
predictions = []


# In[57]:


pip install mlxtend


# In[58]:


from mlxtend.plotting import plot_confusion_matrix


# In[59]:


for i in model:
    predictions.append(i.predict(x_test))
for j in range(7):
    cm = confusion_matrix(y_test, predictions[j])
    plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.summer)
    plt.title(" {} Confusion Matrix".format(model_list[j]))
    plt.xticks(range(2), ["Not Pcos","Pcos"], fontsize=16)
    plt.yticks(range(2), ["Not Pcos","Pcos"], fontsize=16)
    plt.show()


# In[60]:


import pickle
pickle.dump(cat_clf,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))


# In[61]:


x.head()


# In[ ]:




