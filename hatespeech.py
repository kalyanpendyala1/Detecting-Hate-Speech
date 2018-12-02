
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
print(tokenized_tweet.head())
#--------------------------------------------------



#---------------- STEMMING ------------------------
#from nltk.stem import PorterStemmer
from nltk.stem.porter  import *
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])
print(tokenized_tweet.head())
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
print(bow_vectorizer.vocabulary_)
print(bow.shape)
print(bow.toarray())
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

from sklearn.preprocessing import StandardScaler
sc = StandardScaler(with_mean=False)

x_train_std = sc.fit_transform(xtrain_bow)
x_test_std = sc.transform(xvalid_bow)



#--------------Naive Bayes---------------------
from sklearn.naive_bayes  import MultinomialNB
classifier = MultinomialNB()
classifier.fit(xtrain_bow, ytrain)
y_pred = classifier.predict(xvalid_bow)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
print('\n\nResults with naive bayes')
print('Accuracy:%.2f ' %accuracy_score(yvalid,y_pred))
confusion_matrix = confusion_matrix(yvalid,y_pred)
print("Confusion Matrix", confusion_matrix)
print('F1-score', f1_score(yvalid, y_pred, average="macro"))
print('Precision Score: ', precision_score(yvalid, y_pred, average="macro"))
print("recall_score: ",recall_score(yvalid, y_pred, average="macro"))
#-------------------------------------------------



#--------------Decision Tree-------------------
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(xtrain_bow, ytrain)
y_pred = tree.predict(xvalid_bow)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
print('\n\nResults with Decision Tree')
print('Accuracy:%.2f ' %accuracy_score(yvalid,y_pred))
confusion_matrix = confusion_matrix(yvalid,y_pred)
print("Confusion Matrix", confusion_matrix)
print('F1-score', f1_score(yvalid, y_pred, average="macro"))
print('Precision Score: ', precision_score(yvalid, y_pred, average="macro"))
print("recall_score: ",recall_score(yvalid, y_pred, average="macro"))
#-----------------------------------------------


#------------Random Forest----------------------
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=2)
forest.fit(xtrain_bow, ytrain)
y_pred = forest.predict(xvalid_bow)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
print('\n\nResults with Random Forest')
print('Accuracy:%.2f ' %accuracy_score(yvalid,y_pred))
confusion_matrix = confusion_matrix(yvalid,y_pred)
print("Confusion Matrix", confusion_matrix)
print('F1-score', f1_score(yvalid, y_pred, average="macro"))
print('Precision Score: ', precision_score(yvalid, y_pred, average="macro"))
print("recall_score: ",recall_score(yvalid, y_pred, average="macro"))
#-------------------------------------------------


#----------------Logistic Regression--------------
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state=0, solver='lbfgs')
lr.fit(x_train_std, ytrain)
y_pred = lr.predict(x_test_std)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
print('\n\nResults with LogisticRegression')
print('Accuracy:%.2f ' %accuracy_score(yvalid,y_pred))
confusion_matrix = confusion_matrix(yvalid,y_pred)
print("Confusion Matrix", confusion_matrix)
print('F1-score', f1_score(yvalid, y_pred, average="macro"))
print('Precision Score: ', precision_score(yvalid, y_pred, average="macro"))
print("recall_score: ",recall_score(yvalid, y_pred, average="macro"))
#-------------------------------------------------


#----------Support Vector Machines---------------
from sklearn.svm import SVC
#svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10, probability=True) # high precision, low recall, why?
svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10)
svm.fit(x_train_std, ytrain)
y_pred = svm.predict(x_test_std)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
print('\n\nResults with Support Vector Machine')
print('Accuracy:%.2f ' %accuracy_score(yvalid,y_pred))
confusion_matrix = confusion_matrix(yvalid,y_pred)
print("Confusion Matrix", confusion_matrix)
print('F1-score', f1_score(yvalid, y_pred, average="macro"))
print('Precision Score: ', precision_score(yvalid, y_pred, average="macro"))
print("recall_score: ",recall_score(yvalid, y_pred, average="macro"))
#------------------------------------------------


#-------------K-Nearest Neighbors-----------------
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors =5, metric = 'minkowski')
knn.fit(x_train_std, ytrain)
y_pred = knn.predict(x_test_std)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
print('\n\nResults with K-Nearest neighbors')
print('Accuracy:%.2f ' %accuracy_score(yvalid,y_pred))
confusion_matrix = confusion_matrix(yvalid,y_pred)
print("Confusion Matrix", confusion_matrix)
print('F1-score', f1_score(yvalid, y_pred, average="macro"))
print('Precision Score: ', precision_score(yvalid, y_pred, average="macro"))
print("recall_score: ",recall_score(yvalid, y_pred, average="macro"))
#--------------------------------------------------





sys.exit()









