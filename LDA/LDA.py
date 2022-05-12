import warnings, pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
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

    documents = dfNintendoNegatives[dfNintendoNegatives['reviewText'].notna()]['reviewText'].tolist()

    no_topics =  32 #param type: int 
    no_top_words = 5 #param type: int
    no_top_documents = 5 #param type: int

    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tf = tf_vectorizer.fit_transform(documents)
    tf_features_names = tf_vectorizer.get_feature_names()

    lda_model = LatentDirichletAllocation(n_components=no_topics, max_iter=100, learning_method='online', learning_offset=50., random_state=42).fit(tf)
    lda_W = lda_model.transform(tf)
    lda_H = lda_model.components_   #matriz [i,j] probabilidad de instancia i en el cluster j

    print('LDA Topics')
    display_topics(lda_H, lda_W, tf_features_names, documents, no_top_words, no_top_documents)

if __name__=='__main__':
    main()