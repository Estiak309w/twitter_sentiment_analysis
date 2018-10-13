
# coding: utf-8

# In[26]:


import pandas as pd
from sklearn.model_selection import train_test_split


# In[27]:


data1 = pd.read_csv("twitter_dataset/train_tweets.csv")

X_tweet = data1["tweet"]
y =  data1["label"]


# In[28]:


import re
import numpy as np
from sklearn import metrics


# In[29]:


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt

data1['trimmed_tweet'] = np.vectorize(remove_pattern)(X_tweet, "@[\w]*")


# In[30]:


X = data1['trimmed_tweet']


# In[31]:


X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 1)


# In[32]:


from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(stop_words ="english",min_df=2)
vect.fit(X_train,y_train)
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)


# In[33]:



from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train_dtm,y_train)
y_pred_class = nb.predict(X_test_dtm)
accuracy =metrics.accuracy_score(y_test,y_pred_class)
print(accuracy)


# In[10]:


print(metrics.confusion_matrix(y_test,y_pred_class))


# In[21]:


#calculating precission and recall,
# when it predicts a text is straight, it is correct 97 % of the time.
# it correctly identifies 97% of all straight speech.
precission = float(7288/(7288+193))
print(precission)
recall = float(7288/(7288+163))
print(recall)


# In[29]:


y_test.shape
y_test.value_counts()


# In[36]:


# first 10 false positve when actually is straight but predict as hate
X_test[y_test < y_pred_class].head(10)


# In[19]:


X_train_token = vect.get_feature_names()
straight_speech = nb.feature_count_[0,:]
hate_speech = nb.feature_count_[1,:]


# In[22]:


tokens = pd.DataFrame({'straight_speech':straight_speech, 'hate_speech':hate_speech})
print(tokens.shape)
                      


# In[23]:


#count the number of each class
nb.class_count_


# In[22]:


test_data = pd.read_csv("twitter_dataset/test_tweets.csv")
data_2 = test_data["tweet"]


# In[23]:


X['trimmed_tweet_test'] = np.vectorize(remove_pattern)(data_2, "@[\w]*")


# In[24]:


test_data = X["trimmed_tweet_test"]


# In[25]:


test_data.shape


# In[15]:


test_dtm = vect.transform(test_data)
test_predict_class = nb.predict(test_dtm)
print(test_predict_class)


# In[9]:


from sklearn.linear_model import LogisticRegression


# In[12]:


logit = LogisticRegression()
logit.fit(X_train_dtm,y_train)
logit_pred_class = logit.predict(X_test_dtm)
accuracy =metrics.accuracy_score(y_test,logit_pred_class)
print(accuracy)


# In[24]:


# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'alpha': [0.1,0.01, 1]}]
grid_search = GridSearchCV(estimator = nb,
                           param_grid = parameters,
                           scoring = 'accuracy')


# In[26]:


grid_search = grid_search.fit(X_train_dtm, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


# In[27]:


print(best_accuracy)
print(best_parameters)


# In[ ]:




