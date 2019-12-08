import re
import sqlite3

import spacy
import unicodedata
from django.utils.html import strip_tags
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize.toktok import ToktokTokenizer
from textblob import TextBlob

from contractions import CONTRACTION_MAP

nlp = spacy.load('en_core_web_sm', parse=True, tag=True, entity=True)

def multiprocNormalize(dflocal,sender,processName):
    clean_reviews=[]
    # normalize each review in the dataframe
    for i in range(0, len(dflocal)):
        row = dflocal.iloc[i]

        normalize(str(row['reviewText']), clean_reviews)

        """
        if(len(doc0) !=len(doc)):
            print(doc0)
            print(doc)
            print("-------")
        """
        if i % 1000 == 0:
            print(processName + ' Processed ' + str(i) +' / '+str(len(dflocal)))
    dflocal.insert(8, 'Clean_Review', clean_reviews)

    #print(len(dflocal))
    sender.send(dflocal)

    print(processName + ' DONE' )

    return dflocal



def normalize(doc,normDoc):
    tokenizer = ToktokTokenizer()

    doc = strip_html_tags(doc)

    # remove accented characters
    doc = remove_accented_chars(doc)

    # expand contractions
    doc = expand_contractions(doc)

    # remove extra newlines
    doc = re.sub(r'[\r|\n|\r\n]+', '', doc)

    # lemmatize text
    doc = lemmatize_text(doc)

    # remove special characters and\ or digits
    # insert spaces between special characters to isolate them
    special_char_pattern = re.compile(r'([{.(-)!}])')
    doc = special_char_pattern.sub(" \\1 ", doc)
    doc = remove_special_characters(doc, remove_digits=False)
    # remove extra whitespaces
    doc = re.sub(' +', ' ', doc)

    # tokenize and process each token
    tokens = tokenizer.tokenize(doc)
    tokens1 = []
    for token in tokens:
        # remove stopwords
        tokentemp = is_stopword(token)
        if tokentemp is not '':
            # remove repeated characters
            tokens1.append(remove_repeated_characters(tokentemp).lower())

    # bring list back into document
    doc = ' '.join(tokens1)

    # correct spelling
    # doc=correct_spelling(doc)

    normDoc.append(doc)


def parse_text(text, patterns=None):
    """
    delete all HTML tags and entities
    :param text (str): given text
    :param patterns (dict): patterns for re.sub
    :return str: final text

    usage like:
    parse_text('<div class="super"><p>Hello&ldquo;&rdquo;!&nbsp;&nbsp;</p>&lsquo;</div>')
    >>> Hello!
    """
    base_patterns = {
        '&[rl]dquo;': '',
        '&[rl]squo;': '',
        '&nbsp;': ''
    }

    patterns = patterns or base_patterns

    final_text = strip_tags(text)
    for pattern, repl in patterns.items():
        final_text = re.sub(pattern, repl, final_text)
    return final_text


def strip_html_tags(text):
    soup = parse_text(text)
    return soup


def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode(' ascii', 'ignore').decode(' utf-8', 'ignore')
    return text


def expand_contractions(sentence):
    contraction_mapping = CONTRACTION_MAP
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

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


def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text


def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]'
    if not remove_digits:
        pattern = r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text


# correct spelling
def correct_spelling(text):
    return TextBlob(text).correct()


# removing stopwords
stopword_list = set(stopwords.words('english'))


def is_stopword(token):
    token = token.strip()
    if token not in stopword_list:
        return token
    else:
        return ''


# removing repeated words
repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
match_substitution = r'\1\2\3'


def remove_repeated_characters(token):
    if wordnet.synsets(token):
        return token
    else:
        return repeat_pattern.sub(match_substitution, token)
