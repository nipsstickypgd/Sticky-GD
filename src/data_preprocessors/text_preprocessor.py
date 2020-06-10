"""
The code is based on https://miguelmalvarez.com/2016/11/07/classifying-reuters-21578-collection-with-python/
"""
import nltk
from nltk.corpus import stopwords, reuters
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer


def reuters_dataset():
    nltk.download('reuters')
    nltk.download('stopwords')
    stop_words = stopwords.words("english")

    documents = reuters.fileids()

    train_docs_id = [doc for doc in documents if doc.startswith("train")]
    test_docs_id = [doc for doc in documents if doc.startswith("test")]

    train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
    test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]

    print(len(train_docs), len(test_docs))

    vectorizer = TfidfVectorizer(stop_words=stop_words)

    vectorised_train_documents = vectorizer.fit_transform(train_docs)
    vectorised_test_documents = vectorizer.transform(test_docs)

    # print([reuters.categories(doc_id) for doc_id in test_docs_id])

    mlb = MultiLabelBinarizer()
    train_labels = mlb.fit_transform([reuters.categories(doc_id) for doc_id in train_docs_id])
    test_labels = mlb.transform([reuters.categories(doc_id) for doc_id in test_docs_id])

    return vectorised_train_documents.toarray(), vectorised_test_documents.toarray(), train_labels, test_labels
