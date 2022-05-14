import warnings, pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
warnings.filterwarnings('ignore')

#def display_topics(H, W, feature_names, documents, no_top_words, no_top_documents):
#    f = open('LDA/result.txt', 'w')
#    for topic_idx, topic in enumerate(H):
#        print('\nTopic %d:' % (topic_idx))
#        f.write('\n\n\nTopic %d:\n' % (topic_idx))
#        topic_features = []
#        for i in topic.argsort()[:-no_top_words-1:-1]:
#            topic_features.append(feature_names[i])
#        print('Top '+ str(no_top_words) +' topic words: ' + str(topic_features))  
#        f.write('Top '+ str(no_top_words) +' topic words: ' + str(topic_features) + '\n\n')
#        top_doc_indices = np.argsort(W[:,topic_idx])[::-1][0:no_top_documents]
#        topic_docs = []
#        for doc_index in top_doc_indices:
#            topic_docs.append(doc_index)
#            f.write(str(doc_index) +': '+ documents[doc_index] +'\n')
#        print('Top '+ str(no_top_documents) +' documents in topic: ' +str(topic_docs))

	
def display_topics(H, W, feature_names, documents, no_top_words, no_top_documents, clusters, a, b):
    f = open('LDAIntuitPos-nClusters' + str(clusters) + '-alfa' + str(a) + '-beta' + str(b) + '.txt' , 'w')
    for topic_idx, topic in enumerate(H):
        f.write('\n\nTopic %d:\n' % (topic_idx))
        f.write(''.join([' ' +feature_names[i] + ' ' + str(round(topic[i], 5)) #y esto también
                for i in topic.argsort()[:-no_top_words - 1:-1]]))
        top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:no_top_documents]
        docProbArray=np.array2string(np.argsort(W[:,topic_idx]))
        f.write('\n' + docProbArray + '\n')
        howMany=len(docProbArray + '\n')
        f.write("How Many\n")
        f.write(str(howMany) + '\n')
        for doc_index in top_doc_indices:
            f.write(documents[doc_index] + '\n')
    f.close()

def main():
    dfMergedfMeta = []
    #dfMergedfMeta = pd.read_csv('VG-Reviews5AndMetaElecNintSonyMic_v2.csv')
    dfMergedfMeta = pd.read_csv('HRBlockIntuitReviewsTrainDev_vLast7.csv')
    
    dfIntuit = dfMergedfMeta[dfMergedfMeta['brand'].str.contains('Intuit', na=False)]
    dfHR = dfMergedfMeta[dfMergedfMeta['brand'].str.contains('H&R|HR', na=False)]
    
    dfIntuitPositives = dfIntuit[dfIntuit['overall']>3]
    dfIntuitNegatives = dfIntuit[dfIntuit['overall']<=3]
    dfHRPositives = dfHR[dfHR['overall']>3]
    dfHRNegatives = dfHR[dfHR['overall']<=3]
    
    #documents = dfIntuitNegatives[dfIntuitNegatives['reviewText'].notna()]['reviewText'].tolist()
    documents = dfIntuitPositives[dfIntuitPositives['reviewText'].notna()]['reviewText'].tolist()
    #documents = dfHRNegatives[dfHRNegatives['reviewText'].notna()]['reviewText'].tolist()
    #documents = dfHRPositives[dfHRPositives['reviewText'].notna()]['reviewText'].tolist()

    #dfNintendo = dfMergedfMeta[dfMergedfMeta['brand'].str.contains('Nintendo', na=False)]
    #dfSony = dfMergedfMeta[dfMergedfMeta['brand'].str.contains('Sony|PlayStations|Electr', na=False)]
    #dfMicrosoft = dfMergedfMeta[dfMergedfMeta['brand'].str.contains('[M|m]icrosoft', na=False)]

    #dfNintendoPositives = dfNintendo[dfNintendo['overall']>3]
    #dfNintendoNegatives = dfNintendo[dfNintendo['overall']<=3]
    #dfSonyPositives = dfSony[dfSony['overall']>3]
    #dfSonyNegatives = dfSony[dfSony['overall']<=3]
    #dfMicrosoftPositives = dfMicrosoft[dfMicrosoft['overall']>3]
    #dfMicrosoftNegatives = dfMicrosoft[dfMicrosoft['overall']<=3]

    #documents = dfNintendoNegatives[dfNintendoNegatives['reviewText'].notna()]['reviewText'].tolist()
    #documents = dfNintendoPositives[dfNintendoPositives['reviewText'].notna()]['reviewText'].tolist()
    #documents = dfSonyPositives[dfSonyPositives['reviewText'].notna()]['reviewText'].tolist()
    #documents = dfSonyNegatives[dfSonyNegatives['reviewText'].notna()]['reviewText'].tolist()
    #documents = dfMicrosoftPositives[dfMicrosoftPositives['reviewText'].notna()]['reviewText'].tolist()
    #documents = dfMicrosoftNegatives[dfMicrosoftNegatives['reviewText'].notna()]['reviewText'].tolist()
    

    no_topics = 22 #param type: int 
    no_top_words = 5 #param type: int
    no_top_documents = 5 #param type: int
    alfa = 0.9
    beta = 0.9

    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tf = tf_vectorizer.fit_transform(documents)
    tf_features_names = tf_vectorizer.get_feature_names()

    lda_model = LatentDirichletAllocation(n_components=no_topics, doc_topic_prior= alfa, topic_word_prior= beta, max_iter=100, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
    lda_W = lda_model.transform(tf)
    lda_H = lda_model.components_   #matriz [i,j] probabilidad de instancia i en el cluster j

    print('LDA Topics')
    display_topics(lda_H, lda_W, tf_features_names, documents, no_top_words, no_top_documents, no_topics, alfa, beta)

if __name__=='__main__':
    main()