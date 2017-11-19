from gensim import models, corpora
from data_reorgnization import DataOrganizer
import sys
import os

# Set input files and feature type
if len(sys.argv) <= 1:
    output_dir = "BOW vectors"
else:
    if sys.argv[1] == "1":
        output_dir = "BOW vectors"
        n = 1
    else:
        # Make output folders for ngrams
        output_dir = sys.argv[1] + "grams vectors"
        os.makedirs(output_dir)
        n = int(sys.argv[1])

if len(sys.argv) <= 2:
    body_file = "data/testbodyfile.csv"
else:
    body_file = sys.argv[2]

if len(sys.argv) <= 3:
    headline_file = "data/teststancefile.csv"
else:
    headline_file = sys.argv[3]



# Extract data
data_org = DataOrganizer()
raw_data = data_org.prepare_data(body_file, headline_file)

data = []
# Transfer single tokens into n-grams
if n == 1: # if the user wants only single tokens
    data = raw_data
else: # if the user wants n-grams
    for tuple in raw_data:
        ngrams_headline = data_org.token_to_ngrams(tuple[0], n)
        ngrams_body = data_org.token_to_ngrams(tuple[1], n)
        data.append((ngrams_headline, ngrams_body, tuple[2]))

print (data)

all_tokens = []
for tuple in data:
    all_tokens.append(tuple[0])
    all_tokens.append(tuple[1])

# Construct the dictionary of all texts. Each unique word is assigned an ID.
dictionary = corpora.Dictionary(all_tokens)
dictionary.save(output_dir + "/dict.dict")


# Construct a bag-of-word corpus for headlines and a bao-of-word corpus for article bodies
# Headline with index x corresponds to the article body with index x.

# Term Frequency

tf_headline = [dictionary.doc2bow(item[0]) for item in data]
tf_body = [dictionary.doc2bow(item[1]) for item in data]
corpora.MmCorpus.serialize(output_dir + "/tf_headline_corpus.mm", tf_headline)
corpora.MmCorpus.serialize(output_dir + "/tf_body_corpus.mm", tf_body)
stance = open(output_dir + "/stance_output.txt", "w")
for item in data:
    stance.write(str(item[2])+'\n')


# Tfidf

tfidf_hl = models.TfidfModel(tf_headline)
tfidf_headline = tfidf_hl[tf_headline]
corpora.MmCorpus.serialize(output_dir + "/tfidf_headline_corpus.mm", tfidf_headline)

tfidf_bd = models.TfidfModel(tf_body)
tfidf_body = tfidf_bd[tf_body]
corpora.MmCorpus.serialize(output_dir + "/tfidf_body_corpus.mm", tfidf_body)

