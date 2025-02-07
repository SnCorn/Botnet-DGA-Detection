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
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import joblib

def plot_cm(cm, labels):
    percent = (cm*100.0)/numpy.array(numpy.matrix(cm.sum(axis=1)).T)
    print( 'Confusion Matrix Stats' )
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            print( "%s/%s: %.2f%% (%d/%d)" % (label_i, label_j, (percent[i][j]), cm[i][j], cm[i].sum()) )
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    ax.grid(b=False)
    cax = ax.matshow(percent, cmap='coolwarm')
    pylab.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    pylab.xlabel('Predicted')
    pylab.ylabel('True')
    pylab.show()



print('#1 Read The Data: BotnetDGA/dataset/data_exported_7features.csv')

#Num,No,Domainname,Label,Entropy,REAlexa,REConficker,RECryptolocker,REGoz,REMatsnu,RENew_goz,REPushdo,RERamdo,RERovnix,RETinba,REZeus,MinREBotnets,InformationRadius,ClassificationResult,Result,CharLength,LabelBinary,TreeNewFeature,nGramReputation_Alexa
data = pandas.read_csv('BotnetDGA/dataset/data_exported_7features.csv', header=0, encoding='utf-8')
#print(data.head())
# print(data[['Entropy']].min())
# print(data[['REAlexa']].min())
# print(data[['MinREBotnets']].min())
# print(data[['InformationRadius']].min())
# print(data[['Entropy', 'REAlexa', 'MinREBotnets', 'InformationRadius']].describe())
# print(data[['CharLength']].min())
# print(data[['TreeNewFeature']].min())
# print(data[['nGramReputation_Alexa']].min())
# print(data[['CharLength', 'TreeNewFeature', 'nGramReputation_Alexa' ]].describe())


print('#2 Calculate / Prepare The Data (Features Selection)')

X = data[['CharLength', 'TreeNewFeature', 'nGramReputation_Alexa', 'MinREBotnets']].to_numpy()
# xColumns = ['Entropy','REAlexa','REConficker','RECryptolocker','REGoz','REMatsnu','RENew_goz','REPushdo','RERamdo','RERovnix','RETinba','REZeus','MinREBotnets','InformationRadius','CharLength','TreeNewFeature','nGramReputation_Alexa']
# X = data[xColumns].to_numpy()
# print(numpy.where(numpy.isnan(X)))
# print(X.dtype)
# print(X[:5])
y = numpy.array(data['Label'].tolist())
# print(y[:5])
yBinary = numpy.array(data['LabelBinary'].tolist())
# print(yBinary[:5])


# 1. Univariate Selection (chi-squared (chi²) statistical test)
# bestfeatures = SelectKBest(score_func=chi2, k=10)
# fit = bestfeatures.fit(X,yBinary)
# dfscores = pandas.DataFrame(fit.scores_)
# dfcolumns = pandas.DataFrame(xColumns)
# featureScores = pandas.concat([dfcolumns,dfscores],axis=1)
# featureScores.columns = ['Specs','Score']
# print(featureScores.nlargest(10,'Score'))
# featureScores.nlargest(10,'Score').plot(x='Specs', y='Score', kind='barh')
# pyplot.title('Univariate Selection (chi-squared (chi²) statistical test)')
# pyplot.savefig("Chi-Squared.png", dpi=300, papertype='a4')

# 2. Feature Importance (from Tree Based Classifiers)
# model = ExtraTreesClassifier()
# model.fit(X,yBinary)
# print(model.feature_importances_)
# feat_importances = pandas.Series(model.feature_importances_, index=xColumns)
# feat_importances.nlargest(10).plot(kind='barh')
# pyplot.title('Feature Importance (from Tree Based Classifiers)')
# pyplot.savefig("FeatureImportance.png", dpi=300, papertype='a4')
# print(feat_importances)

# 3.Correlation Matrix with Heatmap
# headtmapData = data.drop(columns=['Num','No','Domainname','Label','ClassificationResult','Result'])
# corrmat = headtmapData.corr()
# top_corr_features = corrmat.index
# pyplot.figure(figsize=(20,20))
# g=seaborn.heatmap(headtmapData[top_corr_features].corr(),annot=True,cmap="RdYlGn")
# pyplot.title('Correlation Matrix with Heatmap')
# pyplot.savefig("CorrelationMatrixHeatmap.png", dpi=300, papertype='a4')


print('#3 Run The Algorithms')

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



# openModelFilename = "et_botnet.onnx"

# print("\n\nConvert into ONNX format")
# from skl2onnx import convert_sklearn
# from skl2onnx.common.data_types import FloatTensorType
# initial_type = [('float_input', FloatTensorType([None, 4]))]
# print("\ninitial_type")
# print(initial_type)
# onx = convert_sklearn(clf4, initial_types=initial_type)
# print("\nonx")
# print(onx)
# with open(openModelFilename, "wb") as f:
#     f.write(onx.SerializeToString())
# print("\nDONE")

# print("\n\nCompute the prediction with ONNX Runtime")
# import onnxruntime as rt
# import numpy
# sess = rt.InferenceSession(openModelFilename)
# print("\nsess")
# print(sess)
# input_name = sess.get_inputs()[0].name
# print("\ninput_name")
# print(input_name)
# label_name = sess.get_outputs()[0].name
# print("\nlabel_name")
# print(label_name)
# pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]
# print("\npred_onx")
# print(pred_onx)
# print("\nDONE")



for clf, label in zip([clf1, clf2, clf3, clf4, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes',
                                                       'Extra Tree', 'Ensemble']):
    scores = model_selection.cross_val_score(clf, X_test, y_test, cv=5, scoring='accuracy')
    print("Accuracy: %0.6f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

print('\nX_test')
print(X_test)
y_pred = clf2.predict(X_test)
print('\ny_pred')
print(y_pred)
labels = ['legit', 'dga']
cm = confusion_matrix(y_test, y_pred, labels)
print('\ncm')
print(cm)
plot_cm(cm, labels)