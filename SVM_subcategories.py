# this file implements neural network - Multilayer perceptron
# data input and output portions here are handled by the baseline code provided by http://www.fakenewschallenge.org

import sys
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from utils.system import parse_params, check_version


def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]


    yur = [] # only has related/unreleted labels
    for i in range(len(y)):
        if y[i] == 3:
            yur.append(3)
        else:
            yur.append(0) #0 to indicate related


    yr = [] # only include related entries
    Xr = []
    for i in range(len(y)):
        if y[i] == 3: continue
        yr.append(y[i])
        Xr.append(X[i])

    yr_discussVsAd = []
    for i in range(len(y)):
        if y[i] == 3: continue
        elif y[i] == 2: yr_discussVsAd.append(2)
        else: yr_discussVsAd.append(0) # 0 to indicate both agree and disagree

    Xad = []
    yad_agreeVsdisagree = []
    for i in range(len(y)):
        if y[i] == 3 or y[i] == 2: continue
        yad_agreeVsdisagree.append(y[i])
        Xad.append(X[i])



    """
    X: original X
    y: original y
    yur: y for related/unrelated. still the original size, use 0 to indicate related, 3 unrelated
    Xr: subset of X which only contains related pairs
    yr: subset of y which only contains related pairs
    yr_discussVsAd: the same size as yr. use 0 to indicate agree and disagree, 2 for discuss
    Xad: subset of X which only contains agree and disagree
    yad_agreeVsdisagree: same size as Xad, 0 for agree and 1 for disagree
    
    To seperate only related vs unrelated, use X and yur
    To seperate only discuss vs agree/disagree, use Xr and yr_discussVsAd
    To seperate only agree vs disagree, use Xad and yad_agreeVsdisagree
    """


    return X,y,yur,Xr,yr,yr_discussVsAd,Xad,yad_agreeVsdisagree

if __name__ == "__main__":
    check_version()
    parse_params()

    #Load the training dataset and generate folds
    d = DataSet()
    folds,hold_out = kfold_split(d,n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)


    # Load the competition dataset
    competition_dataset = DataSet("competition_test")
    X_competition, y_competition, y_competition_ur, X_competition_r, y_competition_r, y_competition_r_discussVsAd, X_competition_ad, y_competition_ad_agreeVsdisagree = generate_features(competition_dataset.stances, competition_dataset, "competition")

    Xs = dict()
    ys = dict()
    ys_ur = dict()
    Xs_r = dict()
    ys_r = dict()
    ys_r_discussVsAd = dict()
    Xs_ad = dict()
    ys_ad_agreeVsdisagree = dict()

    # Load/Precompute all features now
    X_holdout,y_holdout, y_holdout_ur, X_holdout_r, y_holdout_r, y_holdout_r_discussVsAd, X_holdout_ad, y_holdout_ad_agreeVsdisagree = generate_features(hold_out_stances,d,"holdout")
    for fold in fold_stances:
        Xs[fold],ys[fold], ys_ur[fold], Xs_r[fold], ys_r[fold], ys_r_discussVsAd[fold], Xs_ad[fold], ys_ad_agreeVsdisagree[fold] = generate_features(fold_stances[fold],d,str(fold))


    best_score = 0
    best_fold = None


    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        y_train_ur = np.hstack(tuple([ys_ur[i] for i in ids]))
        X_train_r = np.vstack(tuple([Xs_r[i] for i in ids]))
        y_train_r = np.hstack(tuple([ys_r[i] for i in ids]))
        y_train_r_discussVsAd = np.hstack(tuple([ys_r_discussVsAd[i] for i in ids]))
        X_train_ad = np.vstack(tuple([Xs_ad[i] for i in ids]))
        y_train_ad_agreeVsdisagree = np.hstack(tuple([ys_ad_agreeVsdisagree[i] for i in ids]))

        # print ("X_train:")
        # print (X_train.shape)
        # print (X_train)
        #
        # print ("y_train:")
        # print (y_train)

        X_test = Xs[fold]
        y_test = ys[fold]

        y_test_ur = ys_ur[fold]
        X_test_r = Xs_r[fold]
        y_test_r = ys_r[fold]
        y_test_r_discussVsAd = ys_r_discussVsAd[fold]
        X_test_ad = Xs_ad[fold]
        y_test_ad_agreeVsdisagree = ys_ad_agreeVsdisagree[fold]

        """
        To seperate only related vs unrelated, use X and yur
        To seperate only discuss vs agree/disagree, use Xr and yr_discussVsAd
        To seperate only agree vs disagree, use Xad and yad_agreeVsdisagree
        """


        # SVM Linear
        clf = LinearSVC()
        clf.fit(X_train, y_train)
        LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
                  intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                  multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
                  verbose=0)


        predicted = [LABELS[int(a)] for a in clf.predict(X_test)]

        # print ('predicted:')
        # print (predicted)
        actual = [LABELS[int(a)] for a in y_test_ur]
        # print ('actual:')
        # print (actual)

        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)

        score = fold_score/max_fold_score

        print("Score for fold "+ str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_fold = clf



    #Run on Holdout set and report the final score on the holdout set
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    actual = [LABELS[int(a)] for a in y_holdout_ur]

    print("Scores on the dev set")
    report_score(actual,predicted)
    print("")
    print("")

    #Run on competition dataset
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_competition)]
    actual = [LABELS[int(a)] for a in y_competition_ur]

    print("Scores on the test set")
    report_score(actual,predicted)
