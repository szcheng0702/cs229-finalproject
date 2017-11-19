import csv
from word_stem import Stemmer
from nltk.util import ngrams

"""
Transfer the two separate body and stance data files into a list of tuples
    Each tuple represents a headline-body-stance pair and includes
    Index 0: A list of preprocessed tokens in the headline
    Index 1: A list of preprocessed tokens in the body
    Index 2: Stance
"""
class DataOrganizer:
    def __init__(self):
        self.stemmer = Stemmer()

    def prepare_data(self, body_file, stance_file):
        """
        Transfer the two data files into a list of tuples
        Each tuple includes
        Index 0: A list of preprocessed tokens in the headline
        Index 1: A list of preprocessed tokens in the body
        Index 2: Stance
        """
        data = self.combine_body_and_stance(body_file, stance_file)
        revised_data = []
        for item in data:
            item = list(item)
            headline = self.stemmer.stem(item[0])
            body = self.tokenize(item[1])

            item[0] = headline
            item[1] = body

            revised_data.append(tuple(item))

        return revised_data

    def tokenize(self, text):
        """
        Transfer a text string into a list of word tokens after preprocessing
        """
        tokens = self.stemmer.stem(text)

        return tokens

    def combine_body_and_stance(self, body_file, stance_file):
        """
        Pair each body and stance and convert all data into a list of tuples.
        Each tuple includes:
        Index 0: a headline string
        Index 1: a body string
        Index 2: stance
        """
        # Extract headlines, article bodies, and stances
        id_body = self.organize_bodies(body_file)
        id_headline_stance = self.organize_stances(stance_file)

        # pair headline, body and stance
        headline_body_stance = []
        for key, value in id_headline_stance.items():
            body = id_body[key]

            # change the label of the stance
            stance = 0
            if value[1] == "unrelated":
                stance = 1
            elif value[1] == "agree":
                stance = 2
            elif value[1] == "disagree":
                stance = 3
            elif value[1] == "discuss":
                stance = 4

            pair = (value[0], body, stance)
            headline_body_stance.append(pair)

        return headline_body_stance

    def organize_bodies(self, filename):
        """
        Extract articles bodies
        """
        id_body = {}
        with open(filename, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                id_body[row['Body ID']] = row['articleBody']
        return id_body

    def organize_stances(self, filename):
        """
        Extract headlines and stances
        """
        id_headline_stance = {}
        with open(filename, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                id_headline_stance[row['Body ID']] = (row['Headline'], row['Stance'])
        return id_headline_stance

    def token_to_ngrams(self, tokens, n):
        """
        Transfer a token list into a list of n-grams stirngs.
        Note: The nltk ngrams method only transfer token list into a list of n-grams tuple which is
        not accepted by gensim.dictionary
        """
        ngrams_tuple_list = ngrams(tokens, n)
        ngrams_str_list = []
        for tuple in ngrams_tuple_list:
            ngrams_str_list.append(" ".join([t for t in tuple]))
        return ngrams_str_list


# test
# do = DataOrganizer()
# do.prepare_data("data/train_bodies.csv", "data/train_stances.csv")

