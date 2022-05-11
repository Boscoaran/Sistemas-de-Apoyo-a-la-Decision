#librerias necesarias: nltk, pandas, xgboost
import getopt
import pickle
import sys
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import nltk
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import warnings
warnings.filterwarnings("ignore")

import subprocess
import sys

def install_requirments():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])

def normalize(d):
    #Lematizador para sacar el lema (estimaremos: estimar) y Stemmer para la raíz (estimar: estim)
    lemmatazier = WordNetLemmatizer()
    ps = PorterStemmer()
    #Pasar a minúsculas, sustituir caracteres extraños, eliminar stopwords y puntuación y dividir frase en palabras
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

    #Eliminar features de índice, categóricas y no útiles (todas menos summary y reviewText)
    features=['price', 'rank', 'vote_y', 'unixReviewTime', 'Postal Code', 'title', 'description', 'details', 'brand', 'category', 'main_cat', 'State', 'Unnamed: 0', 'asin', 'reviewerName_y', 'reviewerID_y', 'Unnamed: 5', 'reviewTime_y', 'Country', 'verified_y']
    for feature in features:
        del dataset[feature]  
    #Usamos los datos ya descargados de nltk para evitar errores
    nltk.data.path.append('nltk_data')
    #Nomalizar las dos columnas que vamos a usar
    print('Normalizando reviewText')
    dataset['reviewText']=normalize(dataset['reviewText'])
    print('Review text normalizado')
    print('Normalizando summary')
    dataset['summary']=normalize(dataset['summary'])
    print('Summary normalizado')
    #Las dos features que vamos a usar se concatenan en una sola columna
    all_features = []
    for i in range(len(dataset['summary'])):
        x = str(dataset['summary'][i]) + ' ' + str(dataset['reviewText'][i])
        all_features.append(x)
    dataset['all_features'] = all_features
    del dataset['reviewText']    
    del dataset['summary']
    #La columna overall se mapea para que pase de [1,2,3,4,5] a [0,1,2]
    target_map={1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
    dataset['__target__'] = dataset['overall'].map(target_map)
    del dataset['overall']
    return(dataset)

def main(argv):
    file = ''
    model = ''
    #Obtener parametros pasados por el usuario
    try:
        opts, args = getopt.getopt(argv, 'f:m:h:', ['help=', 'file=', 'model='])
    except getopt.GetoptError:
        print ('python3 estimador_negativos.py -f <file> -m <model>')
        print ('Importante: ejecutar el programa desde su carpeta, es necesario para acceder a los archivos (requirements.txt, nltk_data...)')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print ('python3 estimador_negativos.py -f <file> -m <model>')
            print ('Importante: ejecutar el programa desde su carpeta, es necesario para acceder a los archivos (requirements.txt, nltk_data...)')
        elif opt in ('-f', '--file'):
            file = arg
        elif opt in ('-m', '--model'):
            model = arg
    if file == '' or model == '':
        print('python3 estimador_negativos.py -f <file> -m <model>')
        print ('Importante: ejecutar el programa desde su carpeta, es necesario para acceder a los archivos (requirements.txt, nltk_data...)')
    else:
        print('Instalando dependencias necesarias')
        install_requirments()
        dataset = pd.read_csv(file)
        print("CSV leido, realizando preproceso...")
        dataset = preprocess(dataset)
        print('Preproceso realizado, estimando...')
        data = dataset['all_features']
        target = dataset['__target__']
        estimar(model, data, target)

def estimar(model, testX, testY):
    tfidf_vect = TfidfVectorizer(vocabulary=pickle.load(open('tfidf_vect.pkl', 'rb')))
    testX_tfidf = tfidf_vect.fit_transform(testX)
    #Cargar modelo
    model = pickle.load(open(model, 'rb'))
    #El modelo predice los resultados de X
    predictions = model.predict(testX_tfidf)
    #Imprimir resultados
    print('\nRESULTADOS:\n')
    print(f1_score(testY, predictions, average='weighted'))
    print(classification_report(testY, predictions))
    print(confusion_matrix(testY, predictions))

if __name__=='__main__':
    main(sys.argv[1:])
