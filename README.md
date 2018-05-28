## Project Title
This is a project module for news multi-classification and spam mail binary-classification with machine learning.

## Prerequisites
python 3.5<br />
sklearn 0.19<br />
pip xgboost<br />
pip pickle<br />
pip jieba<br />

## Data set
1.fetch_20newsgroups:<br />
    sklearn.datasets.fetch_20newsgroups(subset='all')<br />
    The number of the news dataset is up to 18846.<br />

2. Data set of Chinese Spam mail binary-classification.<br />
    Url:https://github.com/hrwhisper/SpamMessage/tree/master/data and some data of mine.<br />
    The number of this set is about 80 000, the proportion of spam and normal mail is 1:10.<br />


## Model Type
Some test are in LogisticRegression, LinearSVM, SGD, Decision_Tree, Naive_Bayes, KNN, RandomForest,<br />
Gradient_boosting, XGBoost.<br />

## Feature selection
In this module, we segment Chinese mail by 'jieba', then convert news(and the mail) to input data just by<br />
extract "word count" and "tf-idf" feature.<br />

## Result
1.fetch_20newsgroups:<br />
After compare with different model, we find the max F-1 is  0.926 with Linear_SVM.<br />
Accuracy of svm Classifier:  0.9262599469496021<br />
            
          precision    recall  f1-score   support
          0       0.92      0.92      0.92       157
          1       0.84      0.90      0.87       207
          2       0.90      0.88      0.89       199
          3       0.81      0.78      0.80       193
          4       0.88      0.88      0.88       218
          5       0.93      0.91      0.92       196
          6       0.89      0.89      0.89       198
          7       0.94      0.94      0.94       205
          8       0.98      0.97      0.97       204
          9       0.96      0.98      0.97       182
         10       0.98      0.99      0.99       187
         11       0.97      0.97      0.97       203
         12       0.93      0.91      0.92       198
         13       0.95      0.96      0.95       210
         14       0.96      0.98      0.97       193
         15       0.94      0.97      0.95       200
         16       0.92      0.97      0.95       173
         17       0.98      0.98      0.98       161
         18       0.96      0.89      0.92       165
         19       0.90      0.82      0.86       121
         avg / total       0.93      0.93      0.93      3770

2. Chinese Spam Mail Binary-classification<br />
we find the maximum F-1 is 0.994775 with class_weight={1:5, 0:1}.<br />
Accuracy of lr Classifier:  0.994775<br />
            
          precision    recall  f1-score   support
          0       1.00      1.00      1.00    144054
          1       0.99      0.96      0.97     15946
          avg / total       0.99      0.99      0.99    160000
