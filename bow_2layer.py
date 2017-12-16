# Two models are used to classify stances
# Model 1 classifies related and unrelated stances
# For related stances, Model 2 classifies whether a stance is agree, disagree or discuss

import numpy as np
from feature_engineering import refuting_features, polarity_features,hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features, bow_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission
from sklearn.neural_network import MLPClassifier
from utils.system import parse_params, check_version
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingClassifier


def generate_features(stances, dataset, name):
    h, b, y = [], [], []

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap." + name + ".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting." + name + ".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity." + name + ".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand." + name + ".npy")

    X = np.c_[X_hand, X_refuting, X_polarity, X_overlap]
    return X, y

def generate_features_second_layer(stances, dataset, name):
    h, b, y = [], [], []

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap." + name + ".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting." + name + ".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity." + name + ".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand." + name + ".npy")
    X_overlap2 = gen_or_load_feats(bow_overlap_features, h, b, "features/bow_overlap." + name + ".npy")

    X = np.c_[X_hand, X_refuting, X_polarity, X_overlap, X_overlap2]
    return X, y


if __name__ == "__main__":
    check_version()
    parse_params()

    # Load the training dataset and generate folds
    d = DataSet()
    folds, hold_out = kfold_split(d, n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d, folds, hold_out)

    # Load the competition dataset
    competition_dataset = DataSet("competition_test")
    X_competition1, y_competition1 = generate_features(competition_dataset.stances, competition_dataset, "competition")
    X_competition2, _ = generate_features_second_layer(competition_dataset.stances, competition_dataset, "competition")

    Xs = dict()
    ys = dict()
    Xs2 = dict()

    # Load/Precompute all features now
    X_holdout1, y_holdout1 = generate_features(hold_out_stances, d, "holdout")
    X_holdout2, _ = generate_features_second_layer(hold_out_stances, d, "holdout")

    for fold in fold_stances:
        Xs[fold], ys[fold] = generate_features(fold_stances[fold], d, str(fold))
        Xs2[fold], _ = generate_features_second_layer(fold_stances[fold], d, str(fold))

    best_score = 0
    best_model1 = None

    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]



        # prepare the training set for model 1
        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train_list = [ys[i] for i in ids]

        y_related = []
        for item in y_train_list:
            current = []
            y_related.append(current)
            for stance in item:
                if stance == 3:
                    current.append(3)
                else:
                    current.append(2)

        y_train = np.hstack(tuple(y_related))

        # Model 1: Classify Related and Unrelated
        model1 = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
        model1.fit(X_train, y_train)


        # prepare the training set for model 2
        x_attitude = []
        y_attitude = []
        X_train2 = np.vstack(tuple([Xs2[i] for i in ids]))
        y_train = np.hstack(tuple(y_train_list))
        for index, item in enumerate(list(y_train)):
            if item != 3:
                y_attitude.append(item)
                x_attitude.append(X_train2[index])

        x_attitude = np.array(x_attitude)
        y_attitude = np.array(y_attitude)


        # Model 2: Classify Agree, Disagree, and Discuss
        # model2 = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
        model2 = MultinomialNB()
        model2.fit(x_attitude, y_attitude)


        # prediction
        X_test = Xs[fold]
        y_test = ys[fold]
        results = model1.predict(X_test) # classify related and unrelated

        # for related: classify agree, disagree, and discuss
        X_test2 = Xs2[fold]# using a different set of features
        for index, item in enumerate(list(results)):
            if item!=3:  # If a stance is not unrelated, make further classification
                results[index] = model2.predict(np.array(X_test2[index]).reshape(1, -1))

        predicted = [LABELS[int(a)] for a in results]
        actual = [LABELS[int(a)] for a in y_test]

        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)

        score = fold_score / max_fold_score

        print("Score for fold " + str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_model1 = model1
            best_model2 = model2

    # Run on Holdout set and report the final score on the holdout set
    h_results = best_model1.predict(X_holdout1)
    for index, item in enumerate(list(h_results)):
        if item != 3:
            h_results[index] = best_model2.predict(np.array(X_holdout2[index]).reshape(1, -1))

    predicted = [LABELS[int(a)] for a in h_results]
    actual = [LABELS[int(a)] for a in y_holdout1]

    print("Scores on the dev set")
    report_score(actual, predicted)
    print("")
    print("")

    # Run on competition dataset
    t_results = best_model1.predict(X_competition1)
    for index, item in enumerate(list(t_results)):
        if item != 3:
            t_results[index] = best_model2.predict(np.array(X_competition2[index]).reshape(1, -1))

    predicted = [LABELS[int(a)] for a in t_results]
    actual = [LABELS[int(a)] for a in y_competition1]

    print("Scores on the test set")
    report_score(actual, predicted)
