import warnings
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
warnings.filterwarnings('ignore')

dfMergedfMeta = []
dfMergedfMeta = pd.read_csv('LDA/VG-Reviews5AndMetaElecNintSonyMic_v2.csv')

dfNintendo = dfMergedfMeta[dfMergedfMeta['brand'].str.contains('Nintendo', na=False)]

dfSony = dfMergedfMeta[dfMergedfMeta['brand'].str.contains('Sony|PlayStations|Electr', na=False)]

dfMicrosoft = dfMergedfMeta[dfMergedfMeta['brand'].str.contains('[M|m]icrosoft', na=False)]

dfNintendoPositives = dfNintendo[dfNintendo['overall']>3]
dfNintendoNegatives = dfNintendo[dfNintendo['overall']<=3]

documents = dfNintendoNegatives[dfNintendoNegatives['reviewText'].notna()]['reviewText'].tolist()

no_topics = 5 #param type: int
no_top_words = 30 #param type: int
no_top_documents = 10 #param type: int

def display_topics(H, W, feature_names, documents, no_top_words, no_top_documents):
    f = open('result.txt', 'w')
    for topic_idx, topic in enumerate(H):
        print('Topic %d:' % (topic_idx))
        f.write('Topic %d:' % (topic_idx))
        print(' '.join([feature_names[i] for i in topic.argsort()[:-no_top_documents]]))
        f.write("ESCAPEEEEEEE")
        f.write(' '.join([feature_names[i] for i in topic.argsort()[:-no_top_documents]]))
        f.write("ESCAPEEEEEEE")
        top_doc_indices = np.argsort(W[:,topic_idx])[::-1][0:no_top_documents]
        for doc_index in top_doc_indices:
            print(documents[doc_index])
            f.write(documents[doc_index])

tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_features_names = tf_vectorizer.get_feature_names()

lda_model = LatentDirichletAllocation(n_components=no_topics, max_iter=100, learning_method='online', learning_offset=50., random_state=42).fit(tf)
lda_W = lda_model.transform(tf)
lda_H = lda_model.components_

print('LDA Topics')
display_topics(lda_H, lda_W, tf_features_names, documents, no_top_words, no_top_documents)

