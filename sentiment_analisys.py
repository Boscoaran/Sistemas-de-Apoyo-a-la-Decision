from importlib.util import module_from_spec
from sre_constants import JUMP
import pandas as pd
import string
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from regex import E
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

target_map={1: '0', 2: '0', 3: '1', 4: '2', 5: '2'}

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

    numeric_features=['price', 'rank', 'vote_y', 'unixReviewTime', 'Postal Code']
    text_features=['title', 'description', 'details']
    categorical_features=['brand', 'category', 'main_cat', 'State']
    index_features=['Unnamed: 0', 'asin', 'reviewerName_y', 'reviewerID_y']
    other_features=['Unnamed: 5', 'reviewTime_y', 'Country', 'verified_y']

    #Eliminar features de índice, categóricas y no útiles
    for feature in categorical_features:
        del dataset[feature]  
    for feature in index_features:
        del dataset[feature]  
    for feature in other_features:
        del dataset[feature]
    for feature in text_features:
        del dataset[feature]    
    for feature in numeric_features:
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
    
    global target_map
    dataset['__target__'] = dataset[t].map(target_map)
    del dataset[t]
    dataset=dataset[~dataset['__target__'].isnull()]
    print('Target creado')
    
    dataset.to_csv(r'datos_reviewText.csv', index=False, header=True)

    return(dataset)


def sentiment_analisys(data, target):
  
    tfidf_vect = TfidfVectorizer()
    tfidf_data = tfidf_vect.fit_transform(data)

    smt = SMOTE(random_state=777, k_neighbors=1)
    smt_data, smt_target = smt.fit_resample(tfidf_data, target)

    trainX, testX, trainY, testY = train_test_split(smt_data, smt_target, random_state=0)
    
    '''
    cv = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    cv_train = cv.fit_transform(trainX)    
    tfidf_train = tfidf_transformer.fit_transform(cv_train)
    cv_test = cv.transform(testX)
    '''

    model = LogisticRegression()
    model.fit(trainX, trainY)
    predicted = model.predict(testX)
    probas = model.predict_proba(testX)
    print(f1_score(testY, predicted, average='macro'))
    print(classification_report(testY, predicted))
    print(confusion_matrix(testY, predicted))

       
if __name__=='__main__':
    f='datos.csv'
    t='overall'
    dataset=pd.read_csv(f)
    #dataset=preprocess(dataset)
    data = dataset['all_features']
    target = dataset['__target__']
    sentiment_analisys(data, target)
  