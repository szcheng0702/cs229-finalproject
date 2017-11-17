import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

"""
# Tokenize a string and stem each word according to Part of Speech grammatical tags

# 1. Tag each word with POS tag
# 2. If a token is a stop word (common and useless words: e.g. a, is, about, as, the) or non-alphanumeric character, remove it
# 3. If a word is a noun/verb/adjective/adverb, use the WordNet Lemmatizer to stem the word
# 4. Else use the SnowBall Stemmer to stem the word
"""

class Stemmer:
    def __init__(self):
        # init stemmers
        self.snowball_stemmer = SnowballStemmer("english")
        self.wordnet_lemmatizer = WordNetLemmatizer()

        # init stopwords
        self.stopwords = set(stopwords.words("English"))

    def stem(self, sentence):
        # tokenize a sentence and attach POS tags
        tokens = nltk.word_tokenize(sentence);
        tag_list = nltk.pos_tag(tokens);

        result = [];
        # transfer POS tags. NLTK and WordNetStemmer use different formats of POS tags
        for s in tag_list:
            tag = s[1];
            pos0 = ' ';
            if tag[0] == 'N':  # Noun: transfer NN to n
                pos0 = 'n'
            elif tag[0] == 'V': # Verb: transfer VBP, VBG... to v
                pos0 = 'v'
            elif tag[0] == 'J': # Adjective: transfer JJ to a
                pos0 = 'a'
            elif tag[0] == 'R': # Adverb: transfer RB to r
                pos0 = 'r'

            if s[0] not in self.stopwords and s[0].isalnum():
                if pos0.isalpha():
                    result.append(self.wordnet_lemmatizer.lemmatize(s[0], pos=pos0));
                else:
                    result.append(self.snowball_stemmer.stem(s[0]));

        return result

# test
# stem = Stemmer()
# print (stem.stem("I am going to get a beautiful flower quickly in US"))
# print (stem.stem("I have worked here for three years. "))
