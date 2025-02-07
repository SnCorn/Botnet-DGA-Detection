import pandas 
import numpy 
import matplotlib.pyplot as pyplot
import matplotlib.pylab as pylab
from sklearn.feature_extraction.text import CountVectorizer
import math
from collections import Counter
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


data = pandas.read_csv('dataset/data_exported_1.8million.csv', header=0, encoding='utf-8') #data_exported.csv


expId = 1

while expId < 17:
    print("\n\nExperiment ID = " + str(expId) + "\n")


    if expId == 1:
        X = data[[ 'CharLength', 'TreeNewFeature', 'nGramReputation_Alexa', ]].to_numpy()
    elif expId == 2:
        X = data[[ 'CharLength', 'TreeNewFeature', 'nGramReputation_Alexa', 'REBenign', ]].to_numpy()
    elif expId == 3:
        X = data[[ 'CharLength', 'TreeNewFeature', 'nGramReputation_Alexa', 'MinREBotnets', ]].to_numpy()
    elif expId == 4:
        X = data[[ 'CharLength', 'TreeNewFeature', 'nGramReputation_Alexa', 'Entropy', ]].to_numpy()
    elif expId == 5:
        X = data[[ 'CharLength', 'TreeNewFeature', 'nGramReputation_Alexa', 'InformationRadius' ]].to_numpy()
    elif expId == 6:
        X = data[[ 'CharLength', 'TreeNewFeature', 'nGramReputation_Alexa', 'REBenign', 'MinREBotnets', ]].to_numpy()
    elif expId == 7:
        X = data[[ 'CharLength', 'TreeNewFeature', 'nGramReputation_Alexa', 'REBenign', 'Entropy', ]].to_numpy()
    elif expId == 8:
        X = data[[ 'CharLength', 'TreeNewFeature', 'nGramReputation_Alexa', 'REBenign', 'InformationRadius' ]].to_numpy()
    elif expId == 9:
        X = data[[ 'CharLength', 'TreeNewFeature', 'nGramReputation_Alexa', 'MinREBotnets', 'Entropy', ]].to_numpy()
    elif expId == 10:
        X = data[[ 'CharLength', 'TreeNewFeature', 'nGramReputation_Alexa', 'MinREBotnets', 'InformationRadius' ]].to_numpy()
    elif expId == 11:
        X = data[[ 'CharLength', 'TreeNewFeature', 'nGramReputation_Alexa', 'Entropy', 'InformationRadius' ]].to_numpy()
    elif expId == 12:
        X = data[[ 'CharLength', 'TreeNewFeature', 'nGramReputation_Alexa', 'REBenign', 'MinREBotnets', 'Entropy', ]].to_numpy()
    elif expId == 13:
        X = data[[ 'CharLength', 'TreeNewFeature', 'nGramReputation_Alexa', 'REBenign', 'MinREBotnets', 'InformationRadius' ]].to_numpy()
    elif expId == 14:
        X = data[[ 'CharLength', 'TreeNewFeature', 'nGramReputation_Alexa', 'REBenign', 'Entropy', 'InformationRadius' ]].to_numpy()
    elif expId == 15:
        X = data[[ 'CharLength', 'TreeNewFeature', 'nGramReputation_Alexa', 'MinREBotnets', 'Entropy', 'InformationRadius' ]].to_numpy()
    else:
        X = data[[ 'CharLength', 'TreeNewFeature', 'nGramReputation_Alexa', 'REBenign', 'MinREBotnets', 'Entropy', 'InformationRadius' ]].to_numpy()



    y = numpy.array(data['Label'].tolist())


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    clf1 = LogisticRegression(random_state=1)
    clf1.fit(X_train, y_train)


    clf2 = RandomForestClassifier(bootstrap=True, max_depth=None, min_samples_leaf=1,
                                  n_estimators=1500, n_jobs=40, oob_score=False,
                                  random_state=1, verbose=1)
    clf2.fit(X_train, y_train)


    clf3 = GaussianNB()
    clf3.fit(X_train, y_train)


    clf4 = ExtraTreesClassifier()
    clf4.fit(X_train, y_train)


    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3), ('etr', clf4)], voting='soft')


    for clf, label in zip([clf1, clf2, clf3, clf4, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes',
                                                           'Extra Tree', 'Ensemble']):
        scores = model_selection.cross_val_score(clf, X_test, y_test, cv=5, scoring='accuracy')
        print("Accuracy: %0.6f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
        
        
    y_pred = clf2.predict(X_test)


    labels = ['benign', 'dga']
    cm = confusion_matrix(y_test, y_pred, labels)
    print("\n")
    print(cm)
    
    
    expId += 1



print("\n\nDONE")
