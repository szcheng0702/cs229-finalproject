# Fake News Detection
Zixian Chai, Sizhu Cheng and Xiaowei Wu 

Author names are listed alphabetically. All authors contribute equally. 

This code accompanies our paper [Fake News Stance Detection](/fakenews.pdf).

## Introduction

In this work, multiple machine learning methods are employed to detect the stance of newspaper headlines 
on their bodies, which can serve as an important indication of content authenticity. 

The stance of a headline is compared to its news body from a data set provided 
by Fake News Challenge ([FNC-1](http://www.fakenewschallenge.org)). Each data instance consists of headline, body and
stance. Each stance is one of the {unrelated, discuss, agree, disagree}. 
An example instance is shown below. 40350, 9622, and 25413 instances are randomly
 selected as the training, dev, and test sets, respectively. 
 Features which are believed to differentiate the stances of the corpus are 
 first extracted out. Multiple learning models are used to predict the 
 stance given a headline/body pair.

Example Data Instance

**Headline:** It Begins: HazMat-Wearing Passenger Spotted At Airport

**Body:** 
Last week we hinted at what was to come as Ebola fears spread across America. Today, we get confirmation. As The Daily Caller reports, one passenger at Dulles International Airport outside Washington, D.C. is apparently not taking any chances. A female passenger dressed in a hazmat suit - complete with a full body gown, mask and gloves - was spotted Wednesday waiting for a flight at the airport.

**Stance:** discuss

The current project uses the [baseline code](https://github.com/FakeNewsChallenge/fnc-1-baseline) provided by FNC-1. 

## Requirements
- [Python 3.6](https://www.python.org/downloads/)
- [NLTK 3.2.5 (Natural Language Processing Toolkit)](http://www.nltk.org)
- [Gensim 3.1.0](https://radimrehurek.com/gensim/)
- [NumPy 1.13.3](http://www.numpy.org)
- [Scikit-Learn](http://scikit-learn.org/stable/)
## Features
Feature-related methods are included in the feature_engineering file. To generate a specific feature for the data set, use the following method:
```python
from feature_engineering import feature_method, gen_or_load_feats
from utils.score import LABELS
import numpy as np

def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_feature = gen_or_load_feats(feature_method, h, b, "features/overlap."+name+".npy")

    X = np.c_[X_feature]
    return X,y
```
List of feature methods:

- **Cosine similarity features:** hand_features
- **Bag-of-word features:** bow_overlap_features
- **Refuting features:** refuting_features
- **Polarity features:** polarity_features

## Models
The following files correspond to different models. To check a model's performance, simply execute the corresponding python file. 
```
$ python model_file.py
```
List of model files:
- **Linear SVM:** SVM-linear.py
- **Softmax:** softmax.py
- **Multinomial Naive Bayes:** multinomial_bayes.py
- **Multilayer Perceptron:** NNM.py
- **2-model combination:** 2layer_model.py
- **3-model combination:** 3layer_model.py

## Model Evaluation
The FNC-1 competition metric is used to evaluate model performance. The metric 
is a weighted accuracy score, with 25% weight on correctly classifying “related”
 stances, which includes “agree”, “disagree” and “discuss”,  and “unrelated” 
 stances,  and 75% weight on correctly classifying three “related” stances. 

#### Example Output:
##### Test Set

|               | agree         | disagree      | discuss       | unrelated     |
|-----------    |-------        |----------     |---------      |-----------    |
|   agree       |    118        |     4         |   1444        |   337         |
| disagree      |    26         |     0         |   383         |   289         |
|  discuss      |    122        |     0         |   3555        |   787         |
| unrelated     |    7          |     0         |   168         |   18174       |

Score: 8711.0 out of 11651.25	(74.76451024568179%)


##### Dev Set

|               | agree         | disagree      | discuss       | unrelated     |
|-----------    |-------        |----------     |---------      |-----------    |
|   agree       |    70         |     0         |    577        |    115         |
| disagree      |    16         |     0         |    126        |    20         |
|  discuss      |    49         |     1         |   1483        |    267        |
| unrelated     |     1         |     0         |    37         |   6860        |

Score: 3460.25 out of 4448.5	(77.78464651005957%)
