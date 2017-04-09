from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import csv

LABEL_LIST = ['1-year-experience-needed', '2-4-years-experience-needed', '5-plus-years-experience-needed', 
			  'associate-needed', 'bs-degree-needed', 'full-time-job', 'hourly-wage', 'licence-needed', 
			  'ms-or-phd-needed', 'part-time-job', 'salary', 'supervising-job']

class JobDescriptionDataset:
	
	raw_X = None
	raw_Y = None
	dt_matrix = None
	term_names = None
	binarized_labels = None
	corr_matrix = None
	label_cooccurrence = None

	def __init__(self, filename):
		'''Reads in data file

		Args:
			filename (str): filename of data
		'''
		try:
			with open(filename, "rb") as f:
				header = f.readline()
				data = f.read()
		except IOError:
			raise IOError("train.tsv and test.tsv should be in ./data/")
			
		data = map(lambda x: x.split("\t"), data.split("\n"))
		if len(data[0]) == 1:
			self.raw_X = [x[0] for x in data]
		elif len(data[0]) == 2:
			self.raw_X = [x[1] for x in data]
			self.raw_Y = [x[0] for x in data]
		else:
			raise Exception('input file incorrectly formatted')


	def setDTMatrix(self, vocab=None):
		'''Builds document-term matrix

		Args:
			vocab (iterable): vocabulary to pass to CountVectorizer
		'''
		vect = CountVectorizer(input = "content", encoding = "ascii", token_pattern = "[0-9a-zA-Z\-]+", 
							   decode_error = "ignore", min_df = .005, max_df = .975, ngram_range = (1, 3),
							   vocabulary=vocab, binary = True)
		self.dt_matrix = vect.fit_transform(self.raw_X)
		self.term_names = vect.get_feature_names()
	

	def setBinarizedLabels(self):
		'''Builds 'document-term' matrix for labels'''
		self.binarized_labels = CountVectorizer(token_pattern = "[0-9a-zA-Z\-]+", binary = True).fit_transform(self.raw_Y)


	def setCorrelationMatrix(self):
		'''Computes correlation between label occurrence and term occurence

		This builds a matrix computing Pearson's R between the binary vectors of label and term occurence
		across all documents. For example, The (i, j) element is the correlation between the i'th label
		and the j'th term.
		'''
		if self.binarized_labels is None:
			self.setBinarizedLabels()
		if self.dt_matrix is None:
			self.setDTMatrix()
		A = self.binarized_labels
		B = self.dt_matrix
		A_m = A - A.mean(0)
		B_m = B - B.mean(0)
		ssA = np.power(A_m, 2).sum(0)
		ssB = np.power(B_m, 2).sum(0)
		self.corr_matrix = np.dot(A_m.T, B_m)/np.sqrt(np.dot(ssA.T, ssB))


	def setLabelCooccurrence(self):
		if self.raw_Y is None:
			raise Exception('no labels for this dataset')
		vect = CountVectorizer(token_pattern = "[0-9a-zA-Z\-]+", binary = True)
		label_dt_matrix = vect.fit_transform(self.raw_Y)
		label_counts = (label_dt_matrix.T * label_dt_matrix).todense()
		self.label_cooccurrence = label_counts * 1./label_counts.diagonal()


	def getRawX(self):
		return self.raw_X

	def getRawY(self):
		return self.raw_Y

	def getDTMatrix(self):
		if self.dt_matrix is None:
			self.setDTMatrix()
		return self.dt_matrix

	def getTermNames(self):
		if self.dt_matrix is None:
			self.setDTMatrix()
		return self.term_names

	def getBinarizedLabels(self):
		if self.binarized_labels is None:
			self.setBinarizedLabels()
		return self.binarized_labels

	def getCorrelationMatrix(self):
		if self.corr_matrix is None:
			self.setCorrelationMatrix()
		return self.corr_matrix

	def getLabelCooccurrence(self):
		if self.label_cooccurrence is None:
			self.setLabelCooccurrence()
		return self.label_cooccurrence


def labelMatrixToString(y):
	y_str = list()
	for row in y:
		str_pred = list()
		for i, elem in enumerate(row):
			if elem == 1:
				str_pred.append(LABEL_LIST[i])
		y_str.append(" ".join(str_pred))
	return y_str


def predStringToFile(pred_str, fn):
	with open(fn, "wb") as f:
		writer = csv.writer(f)
		writer.writerow(["tags"])
		writer.writerows([(p,) if p != "" else " " for p in pred_str])


def score(predicted, actual):
	'''Computes the generalized F1 score

	Args:
		predicted (list of str): list of predicted classes. Each prediction should be space separated string.
		actual (list of str): list of true classes. Each should be space separated string.

	Return:
		float: generalized F1 score as defined in Indeed ML Hackathon (unscaled)

	'''
	predicted_list = [elem.split(" ") if elem != '' else [] for elem in predicted]
	actual_list = [elem.split(" ") if elem != '' else [] for elem in actual]
	pred_mat = MultiLabelBinarizer(classes = LABEL_LIST).fit_transform(predicted_list)
	act_mat = MultiLabelBinarizer(classes = LABEL_LIST).fit_transform(actual_list)
	TP = np.multiply(pred_mat, act_mat).sum()
	FP = np.multiply(pred_mat, 1 - act_mat).sum()
	FN = np.multiply(1 - pred_mat, act_mat).sum()
	P = 1.*TP/(TP + FP)
	R = 1.*TP/(TP + FN)
	S = 2*P*R/(P + R)
	return P, R, S


def prunedProbMatrix(prob_matrix):
	'''Prune predictions to enfore mutually exclusive label constraints

	Args:
		prob_matrix (numpy array): tag probability estimates

	Return:
		numpy array: probability matrix with mutually exclusive classes zeroed out
		except for the max in each group
	'''
	prob_matrix_pruned = np.copy(prob_matrix)
	cooccurence_clusters = [(0,1,2), (3,4,7,8), (6,10), (5,9)]
	for i in range(prob_matrix.shape[0]):
		for c in cooccurence_clusters:
			max_elem = max(prob_matrix[i, np.array(c)])
			for j in c:
				if prob_matrix[i, j] != max_elem:
					prob_matrix_pruned[i, j] = 0
	return prob_matrix_pruned
