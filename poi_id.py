#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from my_functions import calc_metrics, computeFraction, feature_generate, feature_ranking


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'salary',
                 'bonus',
                 'long_term_incentive',
                 'deferred_income',
                 'deferral_payments',
                 'loan_advances',
                 'other',
                 'expenses',
                 'director_fees',
                 'total_payments',
                 'exercised_stock_options',
                 'restricted_stock',
                 'restricted_stock_deferred',
                 'total_stock_value',
                 'from_messages',
                 'to_messages',
                 'from_poi_to_this_person',
                 'from_this_person_to_poi',
                 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers
### Conversao do dicionario em dataframe
df = pd.DataFrame.from_dict(data_dict, orient="index")
df.head()
df.info()

### Coluna "poi" - conversão do tipo de boleano para inteiro
df["poi"] = df["poi"].map({True: 1, False: 0})

### Exclusao da coluna "email_address"
df.drop("email_address", axis=1, inplace=True)

### Substituicao da string "NaN" por Nulo
df = df.replace("NaN", np.nan)

print("\n\nExploração dos Dados")
print("- Numero total de data points: {}".format(df.shape[0]))
print("- Numero de características usadas: {}".format(df.shape[1]))
print("- Numero de POI: {}".format(df[df["poi"] == 1]["poi"].count()))
print("- Numero de non-POI: {}".format(df[df["poi"] == 0]["poi"].count()))
print("- Valores ausentes: \n{}\n\n".format(df.isnull().sum()))

df.isnull().sum().plot(kind='barh', color='blue', figsize=(10,8))
plt.title('Number of Nulls by Features')
plt.xlabel('number of nulls')
plt.ylabel('Features')
plt.show()

df.describe().transpose()

plt.scatter(df["salary"], df["bonus"])
plt.title('Salary x Bonus\ndados originais')
plt.xlabel("salary ($)")
plt.ylabel("bonus ($)")
plt.show()

df.drop("TOTAL", axis=0, inplace=True)

plt.scatter(df["salary"], df["bonus"])
plt.title('Salary x Bonus\nexclusao de TOTAL')
plt.xlabel("salary ($)")
plt.ylabel("bonus ($)")
plt.show()

### Exclusão da coluna "loan_advances"
df.drop("loan_advances", axis=1, inplace=True)
features_list.remove("loan_advances")

### Identificacao e eliminacao das linhas
df.isnull().sum(axis=1).sort_values(ascending=False).head()

print("\n\n")
print(df.loc["LOCKHART EUGENE E"])
print("\n\n")
print(df.loc["THE TRAVEL AGENCY IN THE PARK"])
print("\n\n")

df.drop("LOCKHART EUGENE E", axis=0, inplace=True)
df.drop("THE TRAVEL AGENCY IN THE PARK", axis=0, inplace=True)

### Substituicao de Nulo por 0
df = df.replace(np.nan, 0)


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
### Conversao do dataframe em dicionario
my_dataset = df.to_dict('index')

for name in my_dataset:
    data_point = my_dataset[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    data_point["fraction_from_poi"] = fraction_from_poi

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    data_point["fraction_to_poi"] = fraction_to_poi

### Update features_list with new features
features_list += ["fraction_from_poi", "fraction_to_poi"]

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

plt.scatter(data[:,-2], data[:,-1])
plt.title("fraction_from_poi x fraction_to_poi")
plt.xlabel("fraction_from_poi")
plt.ylabel("fraction_to_poi")
plt.show()

### Select features according to the k highest scores - SelectKBest
from sklearn.feature_selection import SelectKBest

select = SelectKBest(k='all')
select.fit(features, labels)
scores = select.scores_
pvalues = select.pvalues_
scores_list = zip(features_list[1:], scores, pvalues)
scores_list_sort = list(reversed(sorted(scores_list, key=lambda x: x[1])))
print('Features Ranking')
print("Order\tFeature\t\t\t   Score\t   P-Value")
features_kbest = [features_list[0]]
for cont in range(0,len(scores_list_sort)-1):
    print("{}\t{:20}\t {:10.5f}\t{:10.5f}".format(cont+1, scores_list_sort[cont][0], scores_list_sort[cont][1], scores_list_sort[cont][2]))
    features_kbest.append(scores_list_sort[cont][0])
print("\n\n")

### Selecting 13 best features according scores
k = 13
features_kbest = features_kbest[:k+1]


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

# Create and test the Decision Tree Classifier
clf_dtc = DecisionTreeClassifier(random_state=42)

# Create and test the SVM Classifier
clf_svc = SVC(kernel='linear', random_state=42, max_iter=1000, tol=0.001)

# Create and test the AdaBoost Classifier
clf_adb = AdaBoostClassifier(random_state=42)

clfs = [clf_dtc, clf_svc, clf_adb]


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

# Pipeline configurations to Decision Tree Classifier
pipe_dtc = Pipeline([("scaler", StandardScaler()), 
                     ("classifier", clf_dtc)
                     ])
param_dtc = {"scaler": (StandardScaler(), MinMaxScaler(), None), 
             "classifier__criterion": ("gini","entropy"), 
             "classifier__splitter": ("best", "random"), 
             "classifier__max_depth": [2, 3, 4, None], 
             "classifier__min_samples_split": [1.0, 2, 3], 
             "classifier__min_samples_leaf": [0.5, 1, 2],
             "classifier__max_features": ["auto", None],
             "classifier__max_leaf_nodes": [2, 3, 4, None],
             "classifier__class_weight": ("balanced", None)
             }

# Pipeline configurations to SVM Classifier
pipe_svc = Pipeline([("scaler", StandardScaler()), 
                     ("classifier", clf_svc )
                     ])
param_svc = {"scaler": (StandardScaler(), MinMaxScaler(), None), 
             "classifier__C": [0.01, 0.05, 0.1, 0.5, 0.75, 1],
#             "classifier__kernel": ('rbf', 'linear', 'sigmoid'),
#             "classifier__gamma": [0.001, 0.01, 0.1, 0.5, 1, 'auto'], # only rbf, sigmoid or poly kernel
#             "classifier__coef0": [0, 0.05, 0.01, 0.001], # only sigmoid or poly kernel 
             "classifier__shrinking": (False, True), 
             "classifier__probability": (False, True), 
             "classifier__class_weight": ('balanced', None),
             "classifier__decision_function_shape": ('ovo', 'ovr')
             }


# Pipeline configurations to AdaBoost Classifier
pipe_adb = Pipeline([("scaler", StandardScaler()), 
                     ("classifier", clf_adb)
                     ])
param_adb = {"scaler": (StandardScaler(), MinMaxScaler(), None), 
             "classifier__n_estimators": [10, 20, 40, 60, 80], 
             "classifier__learning_rate": [0.05, 0.1, 1.0, 1.5, 1.0, 1.5, 2.0], 
             "classifier__algorithm" : ("SAMME.R", "SAMME")
             }

pipes = [pipe_dtc, pipe_svc, pipe_adb]
parameters = [param_dtc, param_svc, param_adb]

for ind in range(0, len(pipes)):
    clf = GridSearchCV(pipes[ind], parameters[ind], scoring="f1", cv=3, verbose=1)
    
    features_train, features_test, labels_train, labels_test = \
        feature_generate(my_dataset, features_kbest, folds = 1000)

    clf = clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    
    print("\n\nBest Parameters")    
    print(clf.best_params_)

    print("\n\nBest Estimator")    
    print(clf.best_estimator_)
    
    feature_ranking(clf.best_estimator_.named_steps["classifier"], 
                    features_kbest, features_train, labels_train)
    
    try:
        print("\n\nCoefs")  
        print(clf.best_estimator_.named_steps["classifier"].feature_importances_)
    except:  
        print("- coeficientes não disponíveis para exibição")  
        
    print("\n\nTest Classifier")
    test_classifier(clf.best_estimator_, my_dataset, features_kbest)
    print("\n\n")


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

### Modelo adotado DecisionTree
clf = DecisionTreeClassifier(class_weight='balanced', criterion='gini', max_depth=2,
            max_features='auto', max_leaf_nodes=2,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=1.0,
            min_weight_fraction_leaf=0.0, presort=False, random_state=42,
            splitter='best')
features_list = features_kbest
print("Modelo escolhido: DecisionTree")
print(test_classifier(clf, my_dataset, features_list))

dump_classifier_and_data(clf, my_dataset, features_list)