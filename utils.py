import re

import gensim
import nltk
import numpy as np
import spacy
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

from contractions import CONTRACTION_MAP

RNG = 10
np.random.seed(RNG)

from sklearn.decomposition import PCA

from gensim.utils import simple_preprocess


import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42 ## Output Type 3 (Type3) or Type 42 (TrueType)
rcParams['font.sans-serif'] = 'Arial'
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context('talk')

import pickle


COLORS10 = [
'#1f77b4',
'#ff7f0e',
'#2ca02c',
'#d62728',
'#9467bd',
'#8c564b',
'#e377c2',
'#7f7f7f',
'#bcbd22',
'#17becf',
]



# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])
nlp = spacy.load('en_core_web_sm', parse=True, tag=True, entity=True)

# NLTK Stop words
stop_words = set(stopwords.words('english'))

addtl=['from', 'subject', 're', 'edu', 'use', 'would', 'say', 'could',
                   'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some'
                   , 'thank', 'think', 'see', 'rather', 'lot',
                   'make', 'want', 'seem', 'run', 'need', 'even',
                   'right', 'line', 'even', 'also', 'may', 'take', 'come','phone','phones','cellphone','cellphones']
stop_words_updated=stop_words.union(addtl)

not_stopwords = {'no', 'not', 'ne'}
final_stop_words = set([word for word in stop_words_updated if word not in not_stopwords])

#print(final_stop_words)

contraction_mapping = CONTRACTION_MAP
contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                  flags=re.IGNORECASE | re.DOTALL)

FILENAME_PREFIX= './data/amazon_reviews_'

nltk.downloader.download('vader_lexicon')


def writeToDisk(object,name):
    with open(FILENAME_PREFIX+name+'.pickle', "wb") as f:
        pickle.dump(object, f)

def readFromDisk(name):
    f=open(FILENAME_PREFIX+name+'.pickle', "rb")
    return pickle.load(f)



def multiprocNormalize(dflocal, sender, processName):

    print(processName + ' will process ' + str(len(dflocal)))


    def expand_contractions(data):

        def expand_contractions_sentence(sentence):
            def expand_match(contraction):
                match = contraction.group(0)
                first_char = match[0]
                expanded_contraction = contraction_mapping.get(match) \
                    if contraction_mapping.get(match) \
                    else contraction_mapping.get(match.lower())
                expanded_contraction = first_char + expanded_contraction[1:]
                return expanded_contraction

            expanded_sentence = contractions_pattern.sub(expand_match, sentence)
            return expanded_sentence

        data_expanded = [expand_contractions_sentence(review) for review in data]
        return data_expanded

    def sent_to_words(sentences):
        for sent in sentences:
            sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
            sent = re.sub('\s+', ' ', sent)  # remove newline chars
            sent = re.sub("\'", "", sent)  # remove single quotes
            yield (gensim.utils.simple_preprocess(str(sent), deacc=True))  # deacc=True removes punctuations

    # Convert to list
    data = dflocal['reviewText'].values.tolist()

    #tokenize and clean
    data_words = list(sent_to_words(data))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # !python3 -m spacy download en  # run in terminal once
    def process_words(texts, stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):

        """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
        texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
        texts = [bigram_mod[doc] for doc in texts]
        texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
        texts_out = []
        texts_out_str = []
        nlp = spacy.load('en', disable=['parser', 'ner'])
        i = 0
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
            i = i + 1
            if (i % 5000 == 0):
                print(processName + ': lemmatized ' + str(i))
        # remove stopwords once more after lemmatization
        texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]
        for doc in texts_out:
            texts_out_str.append(" ".join([word for word in doc]))
        return texts_out,texts_out_str

    data_ready_tokens,data_ready_str = process_words(data_words,final_stop_words,allowed_postags=['NOUN', 'ADJ', 'ADV'])  # processed Text Data!


    dflocal['Clean_Review'] =data_ready_str
    dflocal['Clean_Review_Tokens']=data_ready_tokens


    # filter rows out that have less than 10 word tokens
    dflocal = dflocal[dflocal['Clean_Review_Tokens'].apply(lambda x: len(x) >= 10)]

    sender.send(dflocal)

    return dflocal


# correct spelling
def correct_spelling(text):
    return TextBlob(text).correct()


# topic model utils ####################################################

def get_topics_terms_weights(weights, feature_names):
    feature_names = np.array(feature_names)
    sorted_indices = np.array([list(row[::-1])
                               for row
                               in np.argsort(np.abs(weights))])
    sorted_weights = np.array([list(wt[index])
                               for wt, index
                               in zip(weights, sorted_indices)])
    sorted_terms = np.array([list(feature_names[row])
                             for row
                             in sorted_indices])

    topics = [np.vstack((terms.T,
                         term_weights.T)).T
              for terms, term_weights
              in zip(sorted_terms, sorted_weights)]

    return topics


def print_topics_gensim(topic_model, total_topics=1,
                        weight_threshold=0.0001,
                        display_weights=False,
                        num_terms=None):
    for index in range(total_topics):
        topic = topic_model.show_topic(index)
        topic = [(word, round(wt, 2))
                 for word, wt in topic
                 if abs(wt) >= weight_threshold]
        if display_weights:
            print('Topic #' + str(index + 1) + ' with weights')
            print(topic[:num_terms] if num_terms else topic)
        else:
            print('Topic #' + str(index + 1) + ' without weights')
            tw = [term for term, wt in topic]
            print(tw[:num_terms] if num_terms else tw)
        print


def print_topics_udf(topics, total_topics=1,
                     weight_threshold=0.0001,
                     display_weights=False,
                     num_terms=None):
    for index in range(total_topics):
        topic = topics[index]
        topic = [(term, float(wt))
                 for term, wt in topic]
        topic = [(word, round(wt, 2))
                 for word, wt in topic
                 if abs(wt) >= weight_threshold]

        if display_weights:
            print('Topic #' + str(index + 1) + ' with weights')
            print(topic[:num_terms] if num_terms else topic)
        else:
            print('Topic #' + str(index + 1) + ' without weights')
            tw = [term for term, wt in topic]
            print(tw[:num_terms] if num_terms else tw)
        print


# below for unsupervised senti analysis ########

def analyze_sentiment_sentiwordnet_lexicon_multiproc(reviews,sender,procIndex):

    i=0
    size=len(reviews)
    sentiments=[]
    sentiments.append(procIndex)
    for review in reviews:
        i=i+1
        sentiment=analyze_sentiment_sentiwordnet_lexicon(review)
        sentiments.append(sentiment)
        if(i%1000 == 0):
            print('process ' + str(procIndex)+' reached '+str(i) + ' of '+ str(size))

    sender.send(sentiments)

    return sentiments

def analyze_sentiment_sentiwordnet_lexicon(review):
    # tokenize and POS tag text tokens
    tagged_text = [(token.text, token.tag_) for token in nlp(review)]
    pos_score = neg_score = token_count = obj_score = 0
    # get wordnet synsets based on POS tags
    # get sentiment scores if synsets are found
    for word, tag in tagged_text:
        ss_set = None
        if 'NN' in tag and list(swn.senti_synsets(word, 'n')):
            ss_set = list(swn.senti_synsets(word, 'n'))[0]
        elif 'VB' in tag and list(swn.senti_synsets(word, 'v')):
            ss_set = list(swn.senti_synsets(word, 'v'))[0]
        elif 'JJ' in tag and list(swn.senti_synsets(word, 'a')):
            ss_set = list(swn.senti_synsets(word, 'a'))[0]
        elif 'RB' in tag and list(swn.senti_synsets(word, 'r')):
            ss_set = list(swn.senti_synsets(word, 'r'))[0]
        # if senti-synset is found
        if ss_set:
            # add scores for all found synsets
            pos_score += ss_set.pos_score()
            neg_score += ss_set.neg_score()
            obj_score += ss_set.obj_score()
            token_count += 1

    # aggregate final scores
    final_score = pos_score - neg_score
    norm_final_score = round(float(final_score) / token_count, 2)
    final_sentiment = 1 if norm_final_score >= 0 else 0


    return final_sentiment


def analyze_sentiment_vader_multiproc(reviews,threshold,sender,procIndex):

    i=0
    size=len(reviews)
    sentiments=[]
    sentiments.append(procIndex)
    for review in reviews:
        i=i+1
        sentiment=analyze_sentiment_vader_lexicon(review,threshold)
        sentiments.append(sentiment)
        if(i%1000 == 0):
            print('process ' + str(procIndex)+' reached '+str(i) + ' of '+ str(size))

    sender.send(sentiments)

    return sentiments

def analyze_sentiment_vader_lexicon(review,threshold):

    # analyze the sentiment for review
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(review)
    # get aggregate scores and final sentiment
    agg_score = scores['compound']
    final_sentiment = 1 if agg_score >= threshold \
        else 0

    return final_sentiment


def pca_plot(X, y):
    pca = PCA(n_components=2)
    X_pc = pca.fit_transform(X)

    fig, ax = plt.subplots()
    mask = y == 0
    ax.scatter(X_pc[mask, 0], X_pc[mask, 1], color=COLORS10[0], label='Class 0', alpha=0.5, s=20)
    ax.scatter(X_pc[~mask, 0], X_pc[~mask, 1], color=COLORS10[1], label='Class 1', alpha=0.5, s=20)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(loc='best')
    return fig