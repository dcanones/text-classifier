from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy
import pandas as pd


punctuation = [",", ":", ";", ".", "'", '"', "’", "?", "/", "-", "+", "&", "(", ")", "!", "¿"]

def clean_text(text, lang):
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def is_empty_string(s):
        if s == '':
            return True
        else:
            return False

    tokenized = word_tokenize(text.lower())
    tokenized_stopfiltered = [w for w in tokenized if w not in stopwords.words(lang)]
    tokenized_stopfiltered_punctfiltered = [(''.join([l for l in word if l not in punctuation])) for word in
                                            tokenized_stopfiltered if not is_number(word)]
    tokenized_stopfiltered_punctfiltered_emptyfiltered = [w for w in tokenized_stopfiltered_punctfiltered if
                                                          not is_empty_string(w)]
    return tokenized_stopfiltered_punctfiltered_emptyfiltered

def get_features(text_clean, possible_features):
    text_words = set(text_clean)
    text_features = dict()
    for word in possible_features:
        text_features['contains({})'.format(word)] = (word in text_words)
    return text_features

def generate_possible_features(data, lang):
    unique_features = []
    single_features = []
    for text in data['text']:
        for word in clean_text(text, lang):
            if word not in unique_features and word not in single_features:
                single_features.append(word)
            elif word not in unique_features and word in single_features:
                unique_features.append(word)
    return unique_features

def train_clf(data, lang):
    possible_features = generate_possible_features(data, lang)
    text_features = [(get_features(clean_text(text, lang), possible_features), category) for [text, category] in
                  data.values]
    classifier = NaiveBayesClassifier.train(text_features)
    return classifier

def test_clf(data, clf, lang):
    possible_features = pd.Series(list(clf._feature_probdist.keys())).str[1].str.extract("\((.*)\)",
                                                                                                     expand=False).values
    text_features = [(get_features(clean_text(text, lang), possible_features), category) for [text, category] in
                     data.values]
    return accuracy(clf, text_features)