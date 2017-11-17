from gensim import models, corpora
from data_reorgnization import DataOrganizer
import sys

# Set input files
if len(sys.argv) <= 1:
    body_file = "data/testbodyfile.csv"
else:
    body_file = sys.argv[1]

if len(sys.argv) <= 2:
    headline_file = "data/teststancefile.csv"
else:
    headline_file = sys.argv[2]


# Extract data
data_org = DataOrganizer()
data = data_org.prepare_data(body_file, headline_file)

all_tokens = []
for tuple in data:
    all_tokens.append(tuple[0])
    all_tokens.append(tuple[1])

# Construct the dictionary of all texts. Each unique word is assigned an ID.
dictionary = corpora.Dictionary(all_tokens)
dictionary.save("BOW vectors/dict.dict")


# Construct a bag-of-word corpus for headlines and a bao-of-word corpus for article bodies
# Headline with index x corresponds to the article body with index x.

# Term Frequency

tf_headline = [dictionary.doc2bow(item[0]) for item in data]
tf_body = [dictionary.doc2bow(item[1]) for item in data]
corpora.MmCorpus.serialize("BOW vectors/tf_headline_corpus.mm", tf_headline)
corpora.MmCorpus.serialize("BOW vectors/tf_body_corpus.mm", tf_body)
stance = open("BOW vectors/stance_output.txt", "w")
for item in data:
    stance.write(str(item[2])+'\n')


# Tfidf

tfidf_hl = models.TfidfModel(tf_headline)
tfidf_headline = tfidf_hl[tf_headline]
corpora.MmCorpus.serialize("BOW vectors/tfidf_headline_corpus.mm", tfidf_headline)

tfidf_bd = models.TfidfModel(tf_body)
tfidf_body = tfidf_bd[tf_body]
corpora.MmCorpus.serialize("BOW vectors/tfidf_body_corpus.mm", tfidf_body)

