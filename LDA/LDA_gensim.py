import warnings, pandas as pd, numpy as np
import gensim.corpora
from gensim.models import LdaModel
warnings.filterwarnings('ignore')

def display_topics(H, W, feature_names, documents, no_top_words, no_top_documents):
    f = open('LDA/result.txt', 'w')
    for topic_idx, topic in enumerate(H):
        print('\nTopic %d:' % (topic_idx))
        f.write('\n\n\nTopic %d:\n' % (topic_idx))
        topic_features = []
        for i in topic.argsort()[:-no_top_words-1:-1]:
            topic_features.append(feature_names[i])
        print('Top '+ str(no_top_words) +' topic words: ' + str(topic_features))  
        f.write('Top '+ str(no_top_words) +' topic words: ' + str(topic_features) + '\n\n')
        top_doc_indices = np.argsort(W[:,topic_idx])[::-1][0:no_top_documents]
        topic_docs = []
        for doc_index in top_doc_indices:
            topic_docs.append(doc_index)
            f.write(str(doc_index) +': '+ documents[doc_index] +'\n')
        print('Top '+ str(no_top_documents) +' documents in topic: ' +str(topic_docs))

def main():
    dfMergedfMeta = []
    dfMergedfMeta = pd.read_csv('LDA/VG-Reviews5AndMetaElecNintSonyMic_v2.csv')

    dfNintendo = dfMergedfMeta[dfMergedfMeta['brand'].str.contains('Nintendo', na=False)]
    dfSony = dfMergedfMeta[dfMergedfMeta['brand'].str.contains('Sony|PlayStations|Electr', na=False)]
    dfMicrosoft = dfMergedfMeta[dfMergedfMeta['brand'].str.contains('[M|m]icrosoft', na=False)]

    dfNintendoPositives = dfNintendo[dfNintendo['overall']>3]
    dfNintendoNegatives = dfNintendo[dfNintendo['overall']<=3]

    documents = str(dfNintendoNegatives[dfNintendoNegatives['reviewText'].notna()]['reviewText'].tolist())

    

    no_topics =  32 #param type: int 
    #no_top_words = 5 #param type: int
    #no_top_documents = 5 #param type: int

    corpus = gensim.corpora.textcorpus.TextCorpus(documents)
    lda_model = LdaModel(corpus=corpus, num_topics=no_topics, alpha='auto', eta='auto')
    
    lda_model.show_topic(10)
    
    #print('LDA Topics')
    #display_topics(lda_H, lda_W, tf_features_names, documents, no_top_words, no_top_documents)

if __name__=='__main__':
    main()