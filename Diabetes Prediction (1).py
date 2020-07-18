#!/usr/bin/env python
# coding: utf-8

# In[47]:


cd 


# In[48]:


cd Desktop


# In[49]:


import pandas as pd
data = pd.read_csv('Diabetes.csv')


# In[50]:


print(data)


# In[51]:


data.info()


# In[52]:


data.describe()


# In[53]:


data.drop('ID',inplace=True,axis=1)


# In[54]:


data.describe()


# In[55]:


print(data)


# Here the above steps represent the DATA COLLECTION.
# The above dataset has 9 Columns namely:
# 1.Pregnancies
# 2.Glucose
# 3.BloodPressure
# 4.SkinThickness
# 5.Insulin
# 6.BMI
# 7.DiabetesPedigreeFunction
# 8.Age
# 9.Outcome
# 
# ---> I have dropped the ID column as it is not necessary

# In[56]:


classes={'No Diabetes':0, 'Diabetic':1}


# In[57]:


data.replace({'Outcome': classes},inplace=True)


# In[58]:


print(data)


# --> The above step is called Label Encoding
# --> I have done label encoding inorder to replace the values of No Diabetes == 0 and Diabetic == 1

# In[59]:


import matplotlib.pyplot as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[60]:


plt.figure(figsize=(12,10))
plt.subplot(221)
sn.distplot(data[data['Outcome']==0].Age, color="red")
plt.title('Age of patients without Diabetes')
plt.subplot(222)
sn.distplot(data[data['Outcome']==1].Age, color="blue")
plt.title('Age of patients with Diabetes')


# In[61]:


data.isnull().sum()


# ---> The above process is Data Cleaning.
# ---> The above graph shows the classification of the patients acccording to age who have Diabetes and who dont have Diabetes.
# ---> The graph above is imported from two libraries matplotlib.plt and seaborn
# ---> In order to make our dataset compatible with machine learning algorithms contained in Sci-kit Learn library, we need to        handle all missing data.
# ---> isnull(). sum(). sum() returns the number of missing values in the data set
# ---> From the above information we can tell that there are no empty values. Now its time to separate the Target Variable
# 

# In[62]:


x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values


# In[63]:


print(x)


# In[64]:


print(y)


# In[65]:


from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.4,random_state=30)


# In[66]:


from sklearn.linear_model import LogisticRegression as logreg
model_logreg = logreg() 
model_logreg.fit(x_train,y_train)


# --> This method is a basic technique in statistical analysis which practicing to predict a data value based on previous observations. This algorithm gives the relationship between a dependent variable and one or more dependent variable.

# In[67]:


y_pred=model_logreg.predict(x_test)


# In[68]:


print(y_pred)


# In[69]:


print(y_test)


# In[70]:


accuracy=model_logreg.score(x_test,y_test)


# In[71]:


print(accuracy)


# In[72]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion_matrix)


# ---> A confusion matrix is a table that is often used to describe the performance of a classification model on a set of test data for which the true values are known.

# In[73]:


Pregnancies=float(input("Enter The Number Of Pregnancies The Patient Had:"))
Glucose=float(input("Enter The Glucose Level Of Patient:"))
BloodPressure=float(input("Enter The Blood Pressure Of The Patient:"))
SkinThickness=float(input("Enter The Skin Thickness Of The Patient:"))
Insulin=float(input("Enter The Insulin Level Of The Patient:"))
BMI=float(input("Enter The Body Mass Index(BMI) Of The Patient: "))
DiabetesPedigreeFunction=float(input("Enter The DiabetesPedigreeFunction Of The Patient: "))
Age=float(input("Enter The Age Of The Patient:"))
X=[[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]
y_p=model_logreg.predict(X)
print()
print("The Patinet is Diabetic If it is 1 AND The Patient is Non Diabetic If it is 0:",y_p)


# In[74]:


sn.pairplot(data)


# In[75]:


sn.pairplot(data,hue='Outcome')


# In[76]:


sn.heatmap(data.corr(),annot=True)


# The above step is Data Vizualization.
# --> PairPlaot = Pairplot is usually a grid of plots for each variable in your dataset. Hence you can quickly see how all the variables are related. This can help to infer which variables are useful, which have skewed distribution etc.
# 
# --> HeatMap = Heatmap is a way to show some sort of matrix plot. To use a heatmap the data should be in a matrix form. By matrix we mean that the index name and the column name must match in some way so that the data that we fill inside the cells are relevant.
