# Indeed Machine Learning Competition
This project details my approach to a multi-label classification problem posed as a competition by Indeed and HackerRank in which I placed 27th out of 248 entrants. See [here](https://www.hackerrank.com/indeed-ml-codesprint-2017) for more details about the competition logistics. The competition itself spanned only one week, but I continued work following the close of submissions. The entirety of that work is presented here.

## The Task
The problem involves tagging job descriptions (unstructured text) with any number of a set of twelve labels such as "bs-degree-needed" and "hourly-wage". For a full description, see [here](https://www.hackerrank.com/contests/indeed-ml-codesprint-2017/challenges/tagging-raw-job-descriptions).

## Index of Files
Below is a description of the various files in this repository.
* **preprocessing.py**: This contains most of the data processing functionality (building document-term matrices, etc) as well as some helper functions like converting a label matrix to a list of strings and writing predictions to a file.
* **exploratory.ipynb**: This notebook contains some exploratory analysis inlcuding label frequency and coocurrence and it generates some simple baseline predictions.
* **models.ipynb**: This notebook implements all of the classifiers.
* **summary.pdf**: This report describes the initial work I did and was required as part of the submission for the contest.

## Baseline Models
I considered a few baseline models to get a feel for the problem and as a benchmark for measuring the improvement of more sophisticated models. In particular, I looked at two methods. The first predicts the top N tags by frequency in the training set for every sample for N = {1, 2, 3, 4}, and the second predicts a tag for a sample if it contains the unigram/bigram/trigram most highly correlated with that tag.

## Feature Engineering
I hypothesized that the presence of terms in a limited set of keywords would be very strong predictors for this problem. I began by generating a binary document-term matrix where the *(i,j)* entry is an indicator of the presence of term *j* in document *i*. The term set included unigrams, bigrams, and trigrams with no stopword removal. Finally, I reduced the dimension of the document-term matrix by including only terms that are highly correlated with some specific tag presence. This matrix acted as my feature input to the classifiers described below. More specifically, I keep only terms that are in the top fifteen correlated terms for any tag.

To assess correlation between a tag *t* and a term *r*, I compute Pearson's R between the *d*-dimensional vectors **Y<sub>t</sub>** and **X<sub>r</sub>** where *d* is the number of documents in the training set. The *i*'th element of **Y<sub>t</sub>** is an indicator of tag *t* being assigned to document *i*, and the *i*'th element of **X<sub>r</sub>** is an indicator of the presence of term *r* in document *i*.

## Classifiers
To evaluate my classifiers and get estimates of test score, I split the labeled dataset into a 70% training set and a 30% validation set. I tested all of my models on the same train/validation split to get comparable results.

I used scikit-learn's OneVsRestClassifier in conjunction with SVC and LogisticRegression. The OVR method works by fitting a separate classifier for each tag against all others. SVC and LogisticRegression produced similar results.

Additionally, I implemented a simple one hidden layer neural network in TensorFlow with ReLU activation on the hidden layer and sigmoid activation on the output layer to naturally account for the multi-label nature of the problem. Using the same inputs, this model very slightly underperformed the others.

## Room for Improvement
I could explore alternative ways of generating features. For example, I could look at TF-IDF scores instead of looking at just a binary document term matrix. Also, I could feed a larger feature set into my models instead of only keeping the most correlated terms, or I could explore other ways of reducing dimensionality. By keeping only the most correlated terms, there is collinearity. The advantage of my approach is that it is very simple and makes the models quick to train, but this potentially comes at the expense of performance. 

Some tags are mutually exclusive (in the training set). For example, "bs-degree-needed" and "associate-needed" never cooccur. See my exploratory analysis for more details. My models do not enforce these constraints, so they could potentially predict both for a given sample. I explored a basic strategy for pruning predictions, but it did not boost performance. I could investigate that further.

Finally, some other things that could be done:
* More model parameter tuning
* Error checking in the training set: A significant number of training samples were not tagged with anything, potentially in error.
* Use cross validation instead of validation set to get better estimates of test error
