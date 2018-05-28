# -*- coding: utf-8 -*-
# author: Wanshan

import os
import sys

import pickle as pkl
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


BASE_DIR = os.path.dirname(__file__)


def lr_init(**kwargs):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(**kwargs)


def sgd_init(**kwargs):
    from sklearn.linear_model import SGDClassifier
    return SGDClassifier(**kwargs)


def decision_tree_init(**kwargs):
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(**kwargs)


def naive_bayes_init(**kwargs):
    from sklearn.naive_bayes import MultinomialNB
    return MultinomialNB(**kwargs)


def svm_init(**kwargs):
    from sklearn.svm import LinearSVC
    return LinearSVC(**kwargs)


def knn_init(**kwargs):
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier(**kwargs)


def random_forest_init(**kwargs):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(**kwargs)


def gradient_boosting_init(**kwargs):
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(**kwargs)


def xgboost_init(**kwargs):
    from xgboost import XGBClassifier
    return XGBClassifier(**kwargs)


class model(object):

    def __init__(self, model_path, model_name, **kwargs):
        self.model_path = os.path.join(BASE_DIR, model_path)
        self.model_name = model_name
        self.kwargs = kwargs

        print(self.model_path)
        if os.path.exists(self.model_path):
            self.model = self.load_model()
            print('1111')
        else:
            self.model = getattr(sys.modules[__name__], '%s_init' % self.model_name.lower())(**self.kwargs)

        print(self.model)

    def train(self, train_data, label_data):
        x_train, x_test, y_train, y_test = train_test_split(train_data,
                                                            label_data,
                                                            test_size=0.2,
                                                            random_state=24)
        print(x_train[:2])
        print(y_train[:2])

        x_train, x_test = self.tf_idf_stop_vector(x_train, x_test)

        # print(x_train, x_test, y_train, y_test)
        # ss = StandardScaler()
        # x_train = ss.fit_transform(x_train)
        # x_test = ss.transform(x_test)

        self.model.fit(x_train, y_train)
        y_predict = self.model.predict(x_test)
        print('Accuracy of %s Classifier: ' % self.model_name, self.model.score(x_test, y_test))
        print(classification_report(y_test, y_predict))

    def inference(self, input_data):
        try:
            print(self.model)
            y_predict = self.model.predict(input_data)
            print('The predict result of %s Classifier are:\n' % self.model_name)
            print(y_predict)
        except:
            print("Check the input_data is fit to model.")
            exit()

    def save_model(self):
        print(self.model_path)
        with open(self.model_path, 'wb') as file:
            pkl.dump(self.model, file)

    def load_model(self):
        try:
            file = open(self.model_path, 'rb')
            model = pkl.load(file)
            return model
        except FileNotFoundError:
            print("Make sure trained_model in model path!")
            raise FileNotFoundError

    @staticmethod
    def count_vector(x_train, x_test):
        count_vec = CountVectorizer()
        x_train = count_vec.fit_transform(x_train)
        x_test = count_vec.transform(x_test)
        return x_train, x_test

    @staticmethod
    def count_stop_vector(x_train, x_test):
        count_stop_vec = CountVectorizer(stop_words='english')
        x_train = count_stop_vec.fit_transform(x_train)
        x_test = count_stop_vec.transform(x_test)
        return x_train, x_test

    @staticmethod
    def tf_idf_vector(x_train, x_test):
        tf_idf_vec = TfidfVectorizer()
        x_train = tf_idf_vec.fit_transform(x_train)
        x_test = tf_idf_vec.transform(x_test)
        return x_train, x_test

    @staticmethod
    def tf_idf_stop_vector(x_train, x_test):
        tf_idf_stop_vec = TfidfVectorizer(stop_words='english')
        x_train = tf_idf_stop_vec.fit_transform(x_train)
        x_test = tf_idf_stop_vec.transform(x_test)
        return x_train, x_test

if __name__ == '__main__':

    news_data = fetch_20newsgroups(subset='all')

    model = model(model_path='model/svm_model.pkl', model_name='svm', penalty='l2')
    print(type(news_data.data))
    model.train(news_data.data, news_data.target)
    model.save_model()

    # x_train, x_test = model.count_stop_vector(test_data, test_data[-20:-1])
    # model = model(model_path='model/test_model.pkl', model_name='svm')
    # prediction = model.inference(input_data)
    # print(prediction)






