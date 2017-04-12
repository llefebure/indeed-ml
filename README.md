# Indeed Machine Learning Competition
This project details my approach to a multi-label classification problem posed as a competition by Indeed and HackerRank. See [here](https://www.hackerrank.com/indeed-ml-codesprint-2017) for more details about the competition logistics. The competition itself spanned only one week which wasn't enough time for me to try all of the methods I wanted to. Therefore, I continued work following the closing of submissions, and the entirety of that work is presented here.

## The Task
The problem involves tagging job descriptions (unstructured text) with any number of a set of twelve labels such as "bs-degree-needed" and "hourly-wage". For a full description, see [here](https://www.hackerrank.com/contests/indeed-ml-codesprint-2017/challenges/tagging-raw-job-descriptions).

## Initial Approach
This section describes the work I did while the competition was live.

### Feature Engineering
I hypothesized that the presence of terms in a limited set of keywords would be very strong predictors for this problem. I began by generating a binary document-term matrix where the *(i,j)* entry is an indicator of the presence of term *j* in document *i*. The term set included unigrams, bigrams, and trigrams with no stopword removal. Finally, I reduced the dimension of the document-term matrix by including only terms that are highly correlated with some specific tag presence. This matrix acted as my feature input to the classifiers described below. More specifically, I keep only terms that are in the top fifteen correlated terms for any tag.

To assess correlation between a tag *t* and a term *r*, I compute Pearson's R between the *d*-dimensional vectors **Y<sub>t</sub>** and **X<sub>r</sub>** where *d* is the number of documents in the training set. The *i*'th element of **Y<sub>t</sub>** is an indicator of tag *t* being assigned to document *i*, and the *i*'th element of **X<sub>r</sub>** is an indicator of the presence of tag *r* in document *i*.
