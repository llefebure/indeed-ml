# Indeed Machine Learning Competition
This project details my approach to a multi-label classification problem posed as a competition by Indeed and HackerRank in which I placed 27th out of 248 entrants. See [here](https://www.hackerrank.com/indeed-ml-codesprint-2017) for more details about the competition logistics. The competition itself spanned only one week, but I continued work following the close of submissions. The entirety of that work is presented here.

## The Task
The problem involves tagging job descriptions (unstructured text) with any number of a set of twelve labels such as "bs-degree-needed" and "hourly-wage". For a full description, see [here](https://www.hackerrank.com/contests/indeed-ml-codesprint-2017/challenges/tagging-raw-job-descriptions).

## Index of Files
Below is a description of the various files in this repository.
* **preprocessing.py**: This contains most of the data processing functionality (building document-term matrices, etc) as well as some helper functions like converting a label matrix to a list of strings and writing predictions to a file.
* **exploratory.ipynb**: This notebook contains some exploratory analysis inlcuding label frequency and coocurrence and it generates some simple baseline predictions.
* **models.ipynb**: This notebook implements SVM and Logistic Regression models.
* **summary.pdf**: This report describes the initial work I did and was required as part of the submission for the contest.

## Initial Approach
This section describes the work I did while the competition was live.

### Feature Engineering
I hypothesized that the presence of terms in a limited set of keywords would be very strong predictors for this problem. I began by generating a binary document-term matrix where the *(i,j)* entry is an indicator of the presence of term *j* in document *i*. The term set included unigrams, bigrams, and trigrams with no stopword removal. Finally, I reduced the dimension of the document-term matrix by including only terms that are highly correlated with some specific tag presence. This matrix acted as my feature input to the classifiers described below. More specifically, I keep only terms that are in the top fifteen correlated terms for any tag.

To assess correlation between a tag *t* and a term *r*, I compute Pearson's R between the *d*-dimensional vectors **Y<sub>t</sub>** and **X<sub>r</sub>** where *d* is the number of documents in the training set. The *i*'th element of **Y<sub>t</sub>** is an indicator of tag *t* being assigned to document *i*, and the *i*'th element of **X<sub>r</sub>** is an indicator of the presence of tag *r* in document *i*.

### Classifiers
I used scikit-learn's OneVsRestClassifier in conjunction with SVC and LogisticRegression. The OVR method works by fitting a separate classifier for each tag against all others. SVC and LogisticRegression produced similar results.
