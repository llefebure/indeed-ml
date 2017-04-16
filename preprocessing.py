from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
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
		'''Builds matrix of label coocurrence'''
		if self.raw_Y is None:
			raise ValueError('no labels for this dataset')
		binarizer = MultiLabelBinarizer()
		label_dt_matrix = binarizer.fit_transform([x.split(" ") for x in self.raw_Y if x != ''])
		self.label_cooccurrence = np.matmul(label_dt_matrix.T, label_dt_matrix)


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
	'''Converts binary label matrix to list of label strings

	Args:
		y (numpy array): binary label matrix
	'''
	y_str = list()
	for row in y:
		str_pred = list()
		for i, elem in enumerate(row):
			if elem == 1:
				str_pred.append(LABEL_LIST[i])
		y_str.append(" ".join(str_pred))
	return y_str


def predStringToFile(pred_str, fn):
	'''Writes prediction string to file to generate submission file
	
	Args:
		pred_str (list of str): prediction strings
		fn (str): filename
	'''
	with open(fn, "wb") as f:
		writer = csv.writer(f)
		writer.writerow(["tags"])
		writer.writerows([(p,) if p != "" else " " for p in pred_str])


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


def score(true, pred, return_values = False):
	p = precision_score(true, pred, average = "micro")
	r = recall_score(true, pred, average = "micro")
	f = f1_score(true, pred, average = "micro")
	if return_values:
		return (p, r, f)
	else:
		return "Precision: %.4f, Recall: %.4f, F1: %.4f" % (p, r, f)
