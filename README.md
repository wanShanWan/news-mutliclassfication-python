## Project Title
This is a project module for news multi-classification in machine learning.
In this module, we convert news to input data just by extract "word count" and "tf-idf" feature.

## Prerequisites
python 3.5\n
sklearn 0.19
pip xgboost
pip pickle

## Data set
sklearn.datasets.fetch_20newsgroups(subset='all')
The number of the news dataset is up to 18846.

## Model Type
Some test are in LogisticRegression, LinearSVM, SGD, Decision_Tree, Naive_Bayes, KNN, RandomForest,
Gradient_boosting, XGBoost.

## Result
After compare with different model, we find the max F-1 is  0.926 with Linear_SVM.

Accuracy of svm Classifier:  0.9262599469496021
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
