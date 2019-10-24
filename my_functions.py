#!/usr/bin/pickle

""" my functions
"""
import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from feature_format import featureFormat, targetFeatureSplit
from sklearn.feature_selection import RFECV

def computeFraction(poi_messages, all_messages):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
        Funcao extraida da Aula 12.4
    """
    fraction = 0.
    if (poi_messages != "NaN") and (all_messages != "NaN"):    
        if (all_messages) != 0: 
            poi_messages = float(poi_messages)
            all_messages = float(all_messages)
            fraction = (poi_messages / all_messages)
    return fraction


def feature_generate(dataset, feature_list, folds = 1000):
    """ genetare features_train, features_test, labels_train, labels_test
        from dataset and feature_list with StratifiedShuffleSplit and 
        MinMaxScaler functions
    """    
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

#    scaler = MinMaxScaler()
#    features = scaler.fit_transform(features)
    
    sss = StratifiedShuffleSplit(labels, folds, test_size=0.3, random_state = 42)    

    for train_idx, test_idx in sss: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
    
    return features_train, features_test, labels_train, labels_test


def calc_metrics(predictions, labels_test):
    """ calculations of evaluation metrics accuracy, precision, recall, f1, f2
        from predictions and labels_test
    """
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0    
    for prediction, truth in zip(predictions, labels_test):
        if prediction == 0 and truth == 0:
            true_negatives += 1
        elif prediction == 0 and truth == 1:
            false_negatives += 1
        elif prediction == 1 and truth == 0:
            false_positives += 1
        elif prediction == 1 and truth == 1:
            true_positives += 1
        else:
            print "Warning: Found a predicted label not == 0 or 1."
            print "All predictions should take value 0 or 1."
            print "Evaluating performance for processed predictions:"
            break    
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print("Accuracy: {:7.5}\tPrecision: {:7.5}\t Recall: {:7.5}\tF1: {:7.5}\tF2: {:7.5}".format(accuracy, precision, recall, f1, f2))
    except:
        print "Got a divide by zero when trying out"
        print "Precision or recall may be undefined due to a lack of true positive predicitons."
    #return accuracy, precision, recall, f1, f2


def feature_ranking(clf, features_kbest, features_train, labels_train):
    """ ranking with recursive feature elimination and cross-validated 
        selection of the best number of features
    """    
    rfecv = RFECV(clf, step=1, cv=3, scoring='f1')
    rfecv.fit(features_train, labels_train)

    plt.figure()
    plt.title('CV Score vs No of Features')
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()

    feature_importance = list(zip(features_kbest, rfecv.support_))
    new_features = []
    for key,value in enumerate(feature_importance):
        if(value[1]) == True:
            new_features.append(value[0])
    
    print("\n\nNew Features")
    print(new_features)
    
    print("\n\nGrid_Scores")
    print(rfecv.grid_scores_)
    
    print("\n\nSupport")
    print(rfecv.support_)
    
    print("\n\nRanking")
    print(rfecv.ranking_)   
