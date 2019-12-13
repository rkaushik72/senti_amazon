import re
import gensim
import numpy as np
import spacy
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from textblob import TextBlob
from contractions import CONTRACTION_MAP

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])
nlp = spacy.load('en_core_web_sm', parse=True, tag=True, entity=True)

# NLTK Stop words
stop_words = stopwords.words('english')
#for email
#stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
stop_words.remove('no')
stop_words.remove('not')

contraction_mapping = CONTRACTION_MAP
contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                  flags=re.IGNORECASE | re.DOTALL)


def multiprocNormalize(dflocal, sender, processName):

    print(processName + ' will process ' + str(len(dflocal)))

    def review_to_words(reviews):
        for review in reviews:
            yield (gensim.utils.simple_preprocess(str(review), deacc=False))  # deacc=True removes punctuations

    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_tokens = []
        texts_str = []

        i=0
        for sent in texts:
            i=i+1
            doc = nlp(" ".join(sent))
            tokens = [token.lemma_ for token in doc if token.pos_ in allowed_postags]
            texts_tokens.append(tokens)
            texts_str.append(' '.join([word for word in tokens]))
            if(i%1000==0):
                print(processName + ': lemmatized '+str(i))

        return texts_tokens, texts_str

    print(processName + ':1/6')

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


    # Convert to list
    data = dflocal['reviewText']
    # Remove new line characters
    data = [re.sub('\s+', ' ', str(review)) for review in data]
    # contractions
    data_expanded = expand_contractions(data)

    print(processName + ':2/6')

    # Remove distracting single quotes
    # data = [re.sub("\'", "", review) for review in data]
    # tokenize each review into words
    data_words = list(review_to_words(data_expanded))
    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    print(processName + ':3/6')

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    # Form Bigrams
    data_words_bigrams = [trigram_mod[bigram_mod[doc]] for doc in data_words_nostops]

    print(processName + ':4/6')

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized_tokens, data_lemmatized_str = lemmatization(data_words_bigrams,
                                                                    allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    print(processName + ':5/6')

    dflocal.insert(8, 'Clean_Review', data_lemmatized_str)
    dflocal.insert(9, 'Clean_Review_Tokens', data_lemmatized_tokens)

    sender.send(dflocal)

    print(processName + ' DONE')

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


