import pickle
import pandas as pd
import string
import nltk
from xgboost import XGBClassifier
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import  train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

def normalize(d):

    lemmatazier = WordNetLemmatizer()
    ps = PorterStemmer()
    for n in range(len(d)):
        s = str(d[n])
        s = s.lower()
        s = s.translate(str.maketrans('', '', string.punctuation))
        s = s.split()
        stop_words=set(stopwords.words('english'))
        filtered_sentence = [w for w in s if not w in stop_words]
        filtered_sentence = []
        for w in s:
            if w not in stop_words and len(w)>1:
                w = lemmatazier.lemmatize(w)
                w = ps.stem(w)
                filtered_sentence.append(w)
        s = filtered_sentence
        d[n]=s
    return (d)


def preprocess(dataset):
    #Eliminar features de índice, categóricas y no útiles
    features=['price', 'rank', 'vote_y', 'unixReviewTime', 'Postal Code', 'title', 'description', 'details', 'brand', 'category', 'main_cat', 'State', 'Unnamed: 0', 'asin', 'reviewerName_y', 'reviewerID_y', 'Unnamed: 5', 'reviewTime_y', 'Country', 'verified_y']
    for feature in features:
        del dataset[feature]
    nltk.data.path.append('nltk_data')
    dataset['reviewText']=normalize(dataset['reviewText'])
    print('Review text normalizado')
    dataset['summary']=normalize(dataset['summary'])
    print('Summary normalizado')
    all_features = []
    for i in range(len(dataset['summary'])):
        x = str(dataset['summary'][i]) + ' ' + str(dataset['reviewText'][i])
        all_features.append(x)
    dataset['all_features']=all_features     
    del dataset['reviewText']    
    del dataset['summary']
    target_map={1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
    dataset['__target__'] = dataset['overall'].map(target_map)
    del dataset['overall']
    return(dataset)

def sentiment_analisys(data, target):
    tfidf_vect = TfidfVectorizer()
    tfidf_data = tfidf_vect.fit_transform(data)
    trainX, testX, trainY, testY = train_test_split(tfidf_data, target, random_state=42, test_size=0.2)
    #smt = SMOTE(random_state=42, k_neighbors=1, sampling_strategy='auto')
    #smt_trainX, smt_trainY = smt.fit_resample(trainX, trainY)
    #modelo = LogisticRegression(C=1, penalty='l2', solver='saga', max_iter=300)
    nest=[1000, 10000]
    max_depth=[8, 10]
    for n in nest:
        for m in max_depth:
            modelo = XGBClassifier(max_depth=n, n_estimators=m)
            modelo.fit(trainX, trainY)
            predicted = modelo.predict(testX)
            print(f1_score(testY, predicted, average='macro'))
            print(classification_report(testY, predicted))
            print(confusion_matrix(testY, predicted))
    
if __name__=='__main__':
    f='datos.csv'
    t='overall'
    dataset=pd.read_csv(f)
    #dataset=preprocess(dataset)
    print('Preprocesado terminado')
    data = dataset['all_features']
    target = dataset['__target__']
    sentiment_analisys(data, target)
  