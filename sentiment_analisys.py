import string
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from numpy import double
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

def normalize(d):
    lemmatazier = WordNetLemmatizer()
    ps = PorterStemmer()
    uniques = np.unique(d)
    uniques_normalized = []
    dic_norm = {}
    for n in range(len(uniques)):
        s = uniques[n]
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
        uniques_normalized.append(s)
        dic_norm[uniques[n]] = uniques_normalized[n]
    for i in range(len(d)):
        d[i]=dic_norm[d[i]]
    return (d)
    

def main():
    f='HRBlockIntuitReviewsTrainDev_vLast7.csv'
    t='overall'

    dataset=pd.read_csv(f)
    del dataset['asin']
    del dataset['Unnamed: 0']
    del dataset['Unnamed: 5']
    del dataset['reviewTime_y']
    del dataset['reviewerName_y']
    del dataset['Country']
    del dataset['rank']
    del dataset['brand']
    del dataset['category']
    del dataset['main_cat']
    del dataset['State']
    del dataset['price']
    del dataset['description']
    del dataset['unixReviewTime']
    del dataset['Postal Code']
    del dataset['details']
    del dataset['vote_y']
    del dataset['verified_y']
    del dataset['reviewerID_y']
    del dataset['reviewText']

    #dataset['price']=dataset['price'].astype(double).round(2)

    text_features=['summary', 'title'] #'title', 'description', 'details', 'reviewText',
    categorical_features=['brand', 'category', 'main_cat', 'State']
    
    nltk.data.path.append('nltk_data')

    for feature in text_features:
        print(dataset['summary'])
        dataset[feature]=normalize(dataset['summary'])

    dic_brand={'Administaff HRTools': 'HR', 'H & R Block': 'HR', 'H&R': 'HR', 'H&R BLCOK': 'HR', 'H&R BLOCK': 'HR', 'H&R Block': 'HR', 'H&amp;R Block': 'HR', 'HRBB9': 'HR', 'Intuit': 'Intuit', 'Intuit Inc.': 'Intuit', 'Intuit Inc./BlueHippo': 'Intuit', 'Intuit, Inc.': 'Intuit', 'John Truby Blockbuster': 'Other', 'Teneron/Block Financial Software': 'Other', 'Video Blocks': 'Other', 'by\n    \n    H&R Block': 'HR', 'by\n    \n    Intuit': 'Intuit'}
    #dataset['brand']=dataset['brand'].map(dic_brand)
    

    target_map={1: '0', 2: '0', 3: '1', 4: '2', 5: '2'}
    dataset['__target__'] = dataset[t].map(target_map)
    del dataset[t]

    print (dataset)

    dataset=dataset[~dataset['__target__'].isnull()]

    train, test = train_test_split(dataset, test_size=0.2, random_state=42, stratify=dataset[['__target__']])
    print('Train:')
    print(train['__target__'].value_counts())
    print("Test: ")
    print(test['__target__'].value_counts())

    trainX = train.drop('__target__', axis=1)
    testX = test.drop('__target__', axis=1)

    trainY = np.array(train['__target__'])
    testY = np.array(test['__target__'])

    clf = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p='1')

    clf.class_weight = 'balanced'
    clf = clf.fit(trainX, trainY)

    predictions = clf.predict(testX)
    probas = clf.predict_proba(testX)
    predictions = pd.Series(data=predictions, index=testX.index, name='predicted_value')
    cols = [
        u'probability_of_value_%s' % label
     for (_, label) in sorted([(int(target_map[label]), label) for label in target_map])
    ]
    probabilities = pd.DataFrame(data=probas, index=testX.index, columns=cols)

    results_test = testX.join(predictions, how='left')
    results_test = results_test.join(probabilities, how='left')
    results_test = results_test.join(test['__target__'], how='left')
    results_test = results_test.rename(columns= {'__target__': 'TARGET'})

    print("F-SCORE:\n")
    print(f1_score(testY, predictions, average=None))
    print("\n\nCLASSIFICATION REPORT:\n")
    print(classification_report(testY,predictions))
    print("\nCONFUSION MATRIX:\n")
    print(confusion_matrix(testY, predictions, labels=[0, 1, 2]))


if __name__=='__main__':
    main()