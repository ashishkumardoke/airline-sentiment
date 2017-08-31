
# coding: utf-8

# In[2]:


import sklearn
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords


# In[3]:


data= pd.read_csv("Tweets.csv")
data.head()


# In[4]:


data.describe


# In[5]:


data.dtypes


# In[6]:


data.shape


# In[7]:


#tweet counts per airline

data.airline.value_counts()


# In[10]:


#Plotting the number of tweets each airlines has received
colors=sns.color_palette("husl", 10) 
pd.Series(data["airline"]).value_counts().plot(kind = "bar",
                        color=colors,figsize=(8,6),fontsize=10,rot = 0, title = "Total No. of Tweets for each Airlines")
plt.xlabel('Airlines', fontsize=10)
plt.ylabel('No. of Tweets', fontsize=10)


# In[11]:


data.airline_sentiment.value_counts()


# In[14]:


colors=sns.color_palette('husl',10)

pd.Series(data["airline_sentiment"]).value_counts().plot(kind = "bar",
                                    color=colors,figsize=(8,6),rot=0, title = "Total No. of Tweets for Each Sentiment")
plt.xlabel('Sentiments', fontsize=10)
plt.ylabel('No. of Tweets', fontsize=10)



# In[15]:


colors=sns.color_palette("husl", 10)
pd.Series(data["airline_sentiment"]).value_counts().plot(kind="pie",colors=colors,
    labels=["negative", "neutral", "positive"],explode=[0.05,0.02,0.04],
    shadow=True,autopct='%.2f', fontsize=12,figsize=(6, 6),title = "Total Tweets for Each Sentiment")


# In[16]:


def plot_sub_sentiment(Airline):
    pdf = data[data['airline']==Airline]
    count = pdf['airline_sentiment'].value_counts()
    Index = [1,2,3]
    color=sns.color_palette("husl", 10)
    plt.bar(Index,count,width=0.5,color=color)
    plt.xticks(Index,['Negative','Neutral','Positive'])
    plt.title('Sentiment Summary of' + " " + Airline)

airline_name = data['airline'].unique()
plt.figure(1,figsize=(12,12))
for i in range(6):
    plt.subplot(3,2,i+1)
    plot_sub_sentiment(airline_name[i])


# In[20]:


data.negativereason.value_counts().head(5)


# In[18]:


#Plotting all the negative reasons 
color=sns.color_palette("husl", 10)
pd.Series(data["negativereason"]).value_counts().plot(kind = "bar",
                        color=color,figsize=(8,6),title = "Total Negative Reasons")
plt.xlabel('Negative Reasons', fontsize=10)
plt.ylabel('No. of Tweets', fontsize=10)


# In[19]:





# In[21]:


color=sns.color_palette("husl", 10)
pd.Series(data["negativereason"]).value_counts().head(5).plot(kind="pie",
                labels=["Customer Service Issue", "Late Flight", "Can't Tell","Cancelled Flight","Lost Luggage"],
                colors=color,autopct='%.2f',explode=[0.05,0,0.02,0.03,0.04],shadow=True,
                fontsize=12,figsize=(6, 6),title="Top 5 Negative Reasons")


# In[23]:


sentiment=pd.crosstab(data.airline, data.airline_sentiment)
sentiment


# In[24]:


percent=sentiment.apply(lambda a: a / a.sum() * 100, axis=1)
percent


# In[25]:


data['tweet_created'] = pd.to_datetime(data['tweet_created'])
data["date_created"] = data["tweet_created"].dt.date


# In[26]:


df = data.groupby(['date_created','airline'])
df = df.airline_sentiment.value_counts()
df.unstack()


# In[28]:


data.head()


# In[29]:


def tweet_to_words(raw_tweet):
    letters_only = re.sub("[^a-zA-Z]", " ",raw_tweet) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] 
    return( " ".join( meaningful_words )) 


# In[30]:


def clean_tweet_length(raw_tweet):
    letters_only = re.sub("[^a-zA-Z]", " ",raw_tweet) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] 
    return(len(meaningful_words)) 



# In[31]:


data['sentiment']=data['airline_sentiment'].apply(lambda x: 0 if x=='negative' else 1)
data.sentiment.head()


# In[32]:


#Splitting the data into train and test
data['clean_tweet']=data['text'].apply(lambda x: tweet_to_words(x))
data['Tweet_length']=data['text'].apply(lambda x: clean_tweet_length(x))
train,test = train_test_split(data,test_size=0.2,random_state=42)


# In[36]:


train_clean_tweet=[]
for tweets in train['clean_tweet']:
    train_clean_tweet.append(tweets)
test_clean_tweet=[]
for tweets in test['clean_tweet']:
    test_clean_tweet.append(tweets)


# In[37]:


from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer(analyzer = "word")
train_features= v.fit_transform(train_clean_tweet)
test_features=v.transform(test_clean_tweet)


# In[35]:


data.columns = data.columns.str.strip()


# In[45]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB


# In[39]:


Classifiers = [
    LogisticRegression(C=0.000000001,solver='liblinear',max_iter=200),
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=200),
    AdaBoostClassifier(),
    GaussianNB()]


# In[40]:


dense_features=train_features.toarray()
dense_test= test_features.toarray()
Accuracy=[]
Model=[]
for classifier in Classifiers:
    try:
        fit = classifier.fit(train_features,train['sentiment'])
        pred = fit.predict(test_features)
    except Exception:
        fit = classifier.fit(dense_features,train['sentiment'])
        pred = fit.predict(dense_test)
    accuracy = accuracy_score(pred,test['sentiment'])
    Accuracy.append(accuracy)
    Model.append(classifier.__class__.__name__)
    print('Accuracy of '+classifier.__class__.__name__+' is '+str(accuracy))  


# In[41]:


logreg = LogisticRegression()

logreg.fit(train_features, train['sentiment'])

