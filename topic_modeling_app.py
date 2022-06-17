import gensim
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet 
stopword_list = nltk.corpus.stopwords.words('english')
lemmitizer = WordNetLemmatizer()
import gensim.corpora as corpora
import streamlit as st
import pyLDAvis.gensim_models
import streamlit.components.v1 as components
import ftfy

from pprint import pprint
lem = WordNetLemmatizer()


wordnet.ensure_loaded()

# def read_text(path):
#     with open(path,'r') as f:
#         text=f.read()
#         text = ftfy.fix_encoding(text)
#     return text



def preprocess(text):
    text = text.lower()
    text = text.strip()
    text = text.replace("\n", '')
    text = text.translate(str.maketrans('','',string.punctuation))
    words = word_tokenize(text)
    text = [w for w in words if w not in stopword_list]
    text = [lem.lemmatize(word) for word in text]
    return text

def corpus(cleaned_text):
    id2word = corpora.Dictionary([cleaned_text])
    print(id2word)
    # Create Corpus
    texts = [cleaned_text]
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    return id2word,corpus

def topicModeling(corpus,id2word):
    # number of topics
    num_topics = 3
    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics)
    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]
    return lda_model



# if __name__ =='__main__':
    # path='E:\Tesseract-OCR-main\Tesseract-OCR-main\Extracted pdf text\extracted_text.txt'
    # inp_text=read_text(path)
    # print(inp_text)
def modeling(text):
    # print("data", text)
    preprocess_text=preprocess(text)
    # print(preprocess_text)
    id2word,corpus=corpus(preprocess_text)
    print("id_to_word",id2word,corpus)

    lda_model=topicModeling(corpus,id2word)

    vis = pyLDAvis.gensim_models.prepare(topic_model=lda_model,
                                  corpus=corpus,
                                  dictionary=id2word)
    # pyLDAvis.enable_notebook()
    # pyLDAvis.display(vis)

    py_lda_vis_html = pyLDAvis.prepared_data_to_html(vis)
    components.html(py_lda_vis_html, width=1300, height=800)

    if st.button('Generate pyLDAvis'):
        with st.spinner('Creating pyLDAvis Visualization ...'):
            py_lda_vis_data = pyLDAvis.gensim_models.prepare(corpus,id2word)
            py_lda_vis_html = pyLDAvis.prepared_data_to_html(py_lda_vis_data)
        with st.expander('pyLDAvis', expanded=True):
            st.markdown('pyLDAvis is designed to help users interpret the topics in a topic model that has been '
                        'fit to a corpus of text data. The package extracts information from a fitted LDA topic '
                        'model to inform an interactive web-based visualization.')
            st.markdown('https://github.com/bmabey/pyLDAvis')
            components.html(py_lda_vis_html, width=1300, height=800)
    # return components.html(py_lda_vis_html, width=1300, height=800)