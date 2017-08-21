from nlp_utils import clean_text, get_features, train_clf, test_clf
from functools import wraps
import pickle
import pandas as pd
import sys

def classifier_required(fun):
    @wraps(fun)
    def class_checker(self):
        if self.classifier is not None:
            return fun(self)
        else:
            print("\nTrain a new classifier before!")
    return class_checker

def data_required(fun):
    @wraps(fun)
    def data_checker(self):
        if self.data is not None:
            return fun(self)
        else:
            print("\nLoad a training set or create a new one before!")
    return data_checker

class TextClassifier(object):
    def __init__(self):
        self.data = None
        self.data_name = None
        self.classifier = None
        self.classifier_name = None
        self.lang = 'english'

    def get_action(self):
        action_text = """
        Please, specify an action:

        1. Load classifier.
        2. Save current classifier.
        3. Load training set.
        4. Save current training set.
        5. Generate new training seg.
        6. Add new examples to current training set.
        7. Train new classifier with current training set.
        8. Make predictions with current classifier.
        9. Test classifier accuracy.
        X. Close TextClassifier.

        Current data: {0}
        Current classifier: {1}

        """.format(self.data_name, self.classifier_name)
        possible_options = ['1','2','3','4', '5', '6', '7', '8', '9', 'X']
        print(action_text)
        option = None
        while option not in possible_options:
            option = input("Please introduce an option: ")
            if option not in possible_options:
                print("\nInput option is not one of the possible choices, please try again")
        return option

    def set_language(self):
        language_text = """
        Please, specify language backend for NLTK:

        1. English.
        2. Spanish.
        """
        possible_options = ['1', '2']
        print(language_text)
        option = None
        while option not in possible_options:
            option = input("Please introduce an option (1, 2): ")
            if option not in possible_options:
                print("\nInput option is not one of the possible choices, please try again")
        if option == 1:
            self.lang = 'english'
        else:
            self.lang = 'spanish'

    def load_classfier(self):
        classifier_name = input("Please specify your classifier filename (without extension): ")
        try:
            with open(classifier_name + '.pickle', 'rb') as f:
                self.classifier = pickle.load(f)
                self.classifier_name = classifier_name
        except FileNotFoundError:
            print('Error. There is no classifier called {}'.format(classifier_name))

    @classifier_required
    def save_classifier_decorator(self):
        classifier_name = self.data_name
        with open(classifier_name + '.pickle', 'wb') as f:
            pickle.dump(self.classifier, f)

    def load_data(self):
        load_data_text = """
        Please specify your training set (without extension). Data must be provided in tsv (tab sepatarated values format):
        """
        data_name = input(load_data_text)
        try:
            self.data = pd.read_csv(data_name + '.csv', sep='\t')
            self.data_name = data_name
        except FileNotFoundError:
            print("Error. There is no data file called {}".format(data_name))

    @data_required
    def save_data(self):
        self.data.to_csv(self.data_name + '.csv', sep='\t', index=False)


    def create_data(self):
        data_name = input("Please write the name of your training set filename (without extension): ")
        self.data_name = data_name
        self.data = pd.DataFrame(columns=['text', 'category'])

    @data_required
    def annotate(self):
        while True:
            text = input("Insert a text (X to exit this mode): ")
            if text == 'X':
                return
            category = input("Insert category: ")
            self.data.loc[len(self.data)] = [text, category]

    @data_required
    def train_classifier(self):
        self.classifier = train_clf(self.data, self.lang)
        self.classifier_name = self.data_name

    @classifier_required
    def make_predictions(self):
        text = None
        while text != 'X':
            text = input("\nIntroduce the text you want to classify (write X to exit this mode): ")
            text_clean = clean_text(text, self.lang)
            possible_features = pd.Series(list(self.classifier._feature_probdist.keys())).str[1].str.extract("\((.*)\)",
                                                                                                             expand=False).values
            text_features = get_features(text_clean, possible_features)
            predicted_category = self.classifier.classify(text_features)
            print("Your text has predicted to belong to category: {}".format(predicted_category))

    @data_required
    @classifier_required
    def test_classifier(self):
        print("\nClassifier accuracy on training set is: {:4.2f}".format(test_clf(self.data, self.classifier, self.lang)))

    def run(self):
        print("Welcome to TextClassifier!")
        self.set_language()
        while True:
            option = self.get_action()
            if option == '1':
                self.load_classfier()
            elif option == '2':
                self.save_classifier_decorator()
            elif option == '3':
                self.load_data()
            elif option == '4':
                self.save_data()
            elif option == '5':
                self.create_data()
            elif option == '6':
                self.annotate()
            elif option == '7':
                self.train_classifier()
            elif option == '8':
                self.make_predictions()
            elif option == '9':
                self.test_classifier()
            elif option == 'X':
                print('Good bye!')
                sys.exit(0)
