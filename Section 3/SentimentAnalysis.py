
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

gmodel = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

def extract_words(sent):
    sent = sent.lower()
    sent = re.sub(r'<[^>]+>', ' ', sent) # strip html tags
    sent = re.sub(r'(\w)\'(\w)', '\1\2', sent) # remove apostrophes
    sent = re.sub(r'\W', ' ', sent) # remove punctuation
    sent = re.sub(r'\s+', ' ', sent) # remove repeated spaces
    sent = sent.strip()
    return sent.split()

# unsupervised training data
import re
import os
unsup_sentences = []

# source: http://ai.stanford.edu/~amaas/data/sentiment/, data from IMDB
for dirname in ["train/pos", "train/neg", "train/unsup", "test/pos", "test/neg"]:
    for fname in sorted(os.listdir("aclImdb/" + dirname)):
        if fname[-4:] == '.txt':
            with open("aclImdb/" + dirname + "/" + fname, encoding='UTF-8') as f:
                sent = f.read()
                words = extract_words(sent)
                unsup_sentences.append(TaggedDocument(words, [dirname + "/" + fname]))

# source: http://www.cs.cornell.edu/people/pabo/movie-review-data/
for dirname in ["review_polarity/txt_sentoken/pos", "review_polarity/txt_sentoken/neg"]:
    for fname in sorted(os.listdir(dirname)):
        if fname[-4:] == '.txt':
            with open(dirname + "/" + fname, encoding='UTF-8') as f:
                for i, sent in enumerate(f):
                    words = extract_words(sent)
                    unsup_sentences.append(TaggedDocument(words, ["%s/%s-%d" % (dirname, fname, i)]))
                
# source: https://nlp.stanford.edu/sentiment/, data from Rotten Tomatoes
with open("stanfordSentimentTreebank/original_rt_snippets.txt", encoding='UTF-8') as f:
    for i, line in enumerate(f):
        words = extract_words(sent)
        unsup_sentences.append(TaggedDocument(words, ["rt-%d" % i]))

import random
class PermuteSentences(object):
    def __init__(self, sents):
        self.sents = sents
        
    def __iter__(self):
        shuffled = list(self.sents)
        random.shuffle(shuffled)
        for sent in shuffled:
            yield sent

permuter = PermuteSentences(unsup_sentences) 
model = Doc2Vec(permuter, dm=0, hs=1, size=50)

# done with training, free up some memory
model.delete_temporary_training_data(keep_inference=True)

model.save('reviews.d2v')
# in other program, we could write: model = Doc2Vec.load('reviews.d2v')

sentences = []
sentvecs = []
sentiments = []
for fname in ["yelp", "amazon_cells", "imdb"]: 
    with open("sentiment labelled sentences/%s_labelled.txt" % fname, encoding='UTF-8') as f:
        for i, line in enumerate(f):
            line_split = line.strip().split('\t')
            sentences.append(line_split[0])
            words = extract_words(line_split[0])
            sentvecs.append(model.infer_vector(words, steps=10)) # create a vector for this document
            sentiments.append(int(line_split[1]))
            
# shuffle sentences, sentvecs, sentiments together
combined = list(zip(sentences, sentvecs, sentiments))
random.shuffle(combined)
sentences, sentvecs, sentiments = zip(*combined)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

clf = KNeighborsClassifier(n_neighbors=9)
clfrf = RandomForestClassifier()

scores = cross_val_score(clf, sentvecs, sentiments, cv=5)
print((np.mean(scores), np.std(scores)))

scores = cross_val_score(clfrf, sentvecs, sentiments, cv=5)
print((np.mean(scores), np.std(scores)))

# bag-of-words comparison
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
pipeline = make_pipeline(CountVectorizer(), TfidfTransformer(), RandomForestClassifier())

scores = cross_val_score(pipeline, sentences, sentiments, cv=5)
print((np.mean(scores), np.std(scores)))

