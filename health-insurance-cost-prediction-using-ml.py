#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd


# In[9]:


data=pd.read_csv("insurance.csv")


# ### 1. Display Top 5 Rows of The Dataset

# In[10]:


data.head()


# ### 2. Check Last 5 Rows of The Dataset

# In[11]:


data.tail()


# ### 3. Find Shape of Our Dataset (Number of Rows And Number of Columns)

# In[12]:


data.shape


# In[13]:


print("Number of Rows",data.shape[0])
print("Number of Columns",data.shape[1])


# ### 4. Get Information About Our Dataset Like Total Number Rows, Total Number of Columns, Datatypes of Each Column And Memory Requirement

# In[14]:


data.info()


# ### 5.Check Null Values In The Dataset

# In[15]:


data.isnull().sum()


# ### 6. Get Overall Statistics About The Dataset

# In[16]:


data.describe(include='all')


# ### 7. Covert Columns From String ['sex' ,'smoker','region' ] To Numerical Values 

# In[17]:


data['sex'].unique()
data['sex']=data['sex'].map({'female':0,'male':1})
data['smoker']=data['smoker'].map({'yes':1,'no':0})
data['region']=data['region'].map({'southwest':1,'southeast':2,
                   'northwest':3,'northeast':4})


# In[18]:


data.head()


# ### 8. Store Feature Matrix In X and Response(Target) In Vector y

# In[19]:


X = data.drop(['charges'],axis=1)
y = data['charges']


# ### 9. Train/Test split
# #### 1. Split data into two part : a training set and a testing set
# #### 2. Train the model(s) on training set
# #### 3. Test the Model(s) on Testing set

# In[20]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# ### 10. Import the models

# In[21]:


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# ### 11. Model Training

# In[22]:


lr = LinearRegression()
lr.fit(X_train,y_train)
svm = SVR()
svm.fit(X_train,y_train)
rf = RandomForestRegressor()
rf.fit(X_train,y_train)
gr = GradientBoostingRegressor()
gr.fit(X_train,y_train)


# ### 12. Prediction on Test Data

# In[23]:


y_pred1 = lr.predict(X_test)
y_pred2 = svm.predict(X_test)
y_pred3 = rf.predict(X_test)
y_pred4 = gr.predict(X_test)

df1 = pd.DataFrame({'Actual':y_test,'Lr':y_pred1,
                  'svm':y_pred2,'rf':y_pred3,'gr':y_pred4})


# In[24]:


df1


# ### 13. Compare Performance Visually 

# In[25]:


import matplotlib.pyplot as plt


# In[26]:


plt.subplot(221)
plt.plot(df1['Actual'].iloc[0:11],label='Actual')
plt.plot(df1['Lr'].iloc[0:11],label="Lr")
plt.legend()

plt.subplot(222)
plt.plot(df1['Actual'].iloc[0:11],label='Actual')
plt.plot(df1['svm'].iloc[0:11],label="svr")
plt.legend()

plt.subplot(223)
plt.plot(df1['Actual'].iloc[0:11],label='Actual')
plt.plot(df1['rf'].iloc[0:11],label="rf")
plt.legend()

plt.subplot(224)
plt.plot(df1['Actual'].iloc[0:11],label='Actual')
plt.plot(df1['gr'].iloc[0:11],label="gr")

plt.tight_layout()

plt.legend()


# ### 14. Evaluating the Algorithm

# In[27]:


from sklearn import metrics


# In[28]:


score1 = metrics.r2_score(y_test,y_pred1)
score2 = metrics.r2_score(y_test,y_pred2)
score3 = metrics.r2_score(y_test,y_pred3)
score4 = metrics.r2_score(y_test,y_pred4)


# In[29]:


print(score1,score2,score3,score4)


# In[30]:


s1 = metrics.mean_absolute_error(y_test,y_pred1)
s2 = metrics.mean_absolute_error(y_test,y_pred2)
s3 = metrics.mean_absolute_error(y_test,y_pred3)
s4 = metrics.mean_absolute_error(y_test,y_pred4)


# In[31]:


print(s1,s2,s3,s4)


# ### 15. Predict Charges For New Customer

# In[32]:


data = {'age' : 40,
        'sex' : 1,
        'bmi' : 40.30,
        'children' : 4,
        'smoker' : 1,
        'region' : 2}


# In[33]:


df = pd.DataFrame(data,index=[0])
df


# In[ ]:





# ### Save Model Using Joblib
# 

# In[34]:


gr = GradientBoostingRegressor()
gr.fit(X,y)


# In[35]:


import joblib


# In[36]:


joblib.dump(gr,'model_joblib_gr')


# In[37]:


model=joblib.load('model_joblib_gr')


# In[38]:


model.predict(df)


# 
# 

# ### GUI
# 

# 

# In[48]:


from tkinter import *


# In[49]:


import joblib


# In[50]:





# In[57]:


def show_entry():
    p1=float(e1.get())
    p2=float(e2.get())
    p3=float(e3.get())
    p4=float(e4.get())
    p5=float(e5.get())
    p6=float(e6.get())
    
    model=joblib.load('model_joblib_gr')
    result=model.predict([[p1,p2,p3,p4,p5,p6]])
    Label(master,text="insurance cost").grid(row=7)
    Label(master,text=result).grid(row=8)

master=Tk()
master.title("Insurance cost prediction")
label = Label(master,text="insurance cost prediction",bg="black",fg="white").grid(row=0,columnspan=2)
Label(master,text="Enter your Age").grid(row=1)
Label(master,text="Male or Female(1/0)").grid(row=2)
Label(master,text="Enter your BMI Value").grid(row=3)
Label(master,text="Enter number of Children").grid(row=4)
Label(master,text="Smoker Yes/No (1/0)").grid(row=5)
Label(master,text="Region (1-4)").grid(row=6)

e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)

e1.grid(row=1,column=1)
e2.grid(row=2,column=1)
e3.grid(row=3,column=1)
e4.grid(row=4,column=1)
e5.grid(row=5,column=1)
e6.grid(row=6,column=1)

Button(master,text="predict",command=show_entry).grid()
mainloop()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




