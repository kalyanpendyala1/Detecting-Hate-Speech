#------------- importing Libraries---------------
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import warnings
import numpy as np
import sys
warnings.filterwarnings("ignore", category=DeprecationWarning)
#---------------------------------------------------


#-------- Loading Data Files ---------------------
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(train.head())
#-------------------------------------------------

#--- combining Test and Train Data files ---------
combi = train.append(test, ignore_index=True,sort=False)
#-------------------------------------------------

#--------- Function to remove Noise ---------------
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt
#--------------------------------------------------
#------------Removing twitter handles--------------
combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*")
#--------------------------------------------------

#Removing special characters, numbers, punctuations
combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]"," ")
#--------------------------------------------------

#----------------Removing Short Words -------------
combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
print(combi.head())
#--------------------------------------------------

#--------------- TOKENIZATION----------------------
tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split())
tokenized_tweet.head()
#--------------------------------------------------



#---------------- STEMMING ------------------------
#from nltk.stem import PorterStemmer
from nltk.stem.porter  import *
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])
tokenized_tweet.head()
#--------------------------------------------------



#-------------- Stiching Tokens Together ----------
for i in range(len(tokenized_tweet)):
	tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
combi['tidy_tweet'] = tokenized_tweet
#--------------------------------------------------

#------------ WordCloud Vizualization -------------
all_words = ' '.join([text for text in combi['tidy_tweet']])

from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
#--------------------------------------------------


#---------Vizualizing non racist tweets------------
normal_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 0]])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
#--------------------------------------------------

#--------- Vizuatlizing Racist Tweets -------------
negative_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 1 ]])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
#--------------------------------------------------


#------------- Function to collect HASHTAGS ------
def hashtag_exract(x):
	hashtags = []
		#looing over all the words in the tweet
	for i in x:
		ht = re.findall(r"#(\w+)",i)
		hashtags.append(ht)
	return hashtags
#-------------------------------------------------



#------ Extracting hashtags from non racist tweets
HT_regular = hashtag_exract(combi['tidy_tweet'][combi['label'] == 0])


#------ Extracting hashtags from racist tweets----
HT_negative = hashtag_exract(combi['tidy_tweet'][combi['label'] == 1])

#---------------unnesting list--------------------
HT_regular = sum(HT_regular, [])
HT_negative = sum(HT_negative, [])
#-------------------------------------------------

a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()), 'Count': list(a.values())})

#---- Top 10 most frequent Hashtags---------------
d = d.nlargest(columns='Count', n=10)
plt.figure(figsize=(16,5))
ax=sns.barplot(data=d, x="Hashtag", y="Count")
ax.set(ylabel = 'Count')
plt.show()

#-------------------------------------------------

#-------Top 10 Racist Hashtags-------------------
b = nltk.FreqDist(HT_negative)
e = pd.DataFrame({})
d = d.nlargest(columns='Count', n=10)
plt.figure(figsize=(16,5))
ax=sns.barplot(data=d, x="Hashtag", y="Count")
ax.set(ylabel = 'Count')
plt.show()

#-----------------------------------------------


#-Feature Extraction using Bag-of-Words approach
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
#-----------------------------------------------

#---------------Feature Matrix------------------
bow = bow_vectorizer.fit_transform(combi['tidy_tweet'])
#-----------------------------------------------


#-----------------------------------------------
#---Builing Models using Logistic Regression----
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
train_bow = bow[:31962, :]
test_bow = bow[31962:,:]

#splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow,train['label'],random_state=42, test_size=0.3)

lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain) #training model
prediction = lreg.predict_proba(xvalid_bow) #predicting on the validation set
prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than equal to 0.3 than 1 else 0
prediction_int = prediction_int.astype(np.int)
print("\n\nF-1 score",f1_score(yvalid, prediction_int,"\n")) #calculating f1-score
#---------------------------------------------------

#------ Using this model to predict Test data-------
test_pred = lreg.predict_proba(test_bow)
y_pred = lreg.predict(xvalid_bow)
print('Accuracy Score: {:.2f}'.format(lreg.score(xvalid_bow, yvalid)))

test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test['label'] = test_pred_int
submission = test[['id', 'label']]
submission.to_csv('sub_lreg_bow.csv', index=False) #writing data to a CSV file
#----------------------------------------------------

#---------------Confusion Matrix----------------------
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(yvalid, y_pred)
print("\nConfusion Matrix", confusion_matrix)
#-----------------------------------------------------


#--Precison, recall, F-measure and support report-----
from sklearn.metrics import classification_report
print('\n',classification_report(yvalid, y_pred))
#-----------------------------------------------------

  

#--------------------Roc Curve-----------------------
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(yvalid, lreg.predict(xvalid_bow))
fpr, tpr, thresholds = roc_curve(yvalid, lreg.predict_proba(xvalid_bow)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
#----------------------------------------------------

sys.exit()
