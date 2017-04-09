import preprocessing as pp
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import numpy as np

# build data sets
train = pp.JobDescriptionDataset("./data/train.tsv")
test = pp.JobDescriptionDataset("./data/test.tsv")

dt_matrix = train.getDTMatrix()
corr_matrix = train.getCorrelationMatrix()
corr_matrix_sort = (-corr_matrix).argsort()
term_names = train.getTermNames()
top_terms = np.array(list(set(corr_matrix_sort[:, :15].flatten().tolist()[0])))

# final training data
X = train.getDTMatrix()
X = X[:, top_terms]
y = train.getBinarizedLabels()

# processed test data
test.setDTMatrix(vocab = np.array(train.getTermNames())[top_terms])
X_test = test.getDTMatrix()

# model building
ovr_svm = OneVsRestClassifier(SVC(kernel = 'linear', probability = True, random_state = 20))
ovr_svm.fit(X, y)
pred_matrix = ovr_svm.predict_proba(X_test) > .235
pred_str = pp.labelMatrixToString(pred_matrix.astype(int))
pp.predStringToFile(pred_str, "./tags.tsv")