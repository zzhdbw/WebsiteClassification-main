# -*- coding: utf-8 -*-
import os.path

import pandas as pd
import numpy as np
import random
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import svm
from DataUtils import getTokens, modelfile_path, vectorfile_path

cachePath = "data/train.pkl"

def getDataFromFile(filename):
    with open(filename, "r", encoding="utf8") as f:
        inputurls, y = [], []
        for line in f:
            link, label = line.strip().split(",")
            inputurls.append(link)
            y.append(label)
    print("read ok!")
    return inputurls, y

# 保存模型及特征
def saveModel(model, vector):
    # 保存模型
    file1 = modelfile_path
    with open(file1, 'wb') as f:
        pickle.dump(model, f)
    f.close()

    # 保存特征
    file2 = vectorfile_path
    with open(file2, 'wb') as f2:
        pickle.dump(vector, f2)
    f2.close()

def train(datapath):
    all_urls, y = getDataFromFile(datapath)
    url_vectorizer = TfidfVectorizer(tokenizer = getTokens)
    x = url_vectorizer.fit_transform(all_urls)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier() #决策树
    # model = RandomForestClassifier(n_estimators=100)#随机森林
    # model = AdaBoostClassifier(n_estimators=10)
    # model = GradientBoostingClassifier()
    # model = LogisticRegression() #逻辑回归
    # model = svm.LinearSVC() #SVM

    model.fit(x_train, y_train)
    # svm_score = svmModel.score(x_test, y_test)
    # print("score: {0:.2f} %".format(100 * svm_score))
    predict = model.predict(x_test)

    p = precision_score(predict, y_test, average="macro")
    r = recall_score(predict, y_test, average="macro")
    f1 = f1_score(predict, y_test, average="macro")
    print(p, r, f1)
    return model, url_vectorizer

if __name__ == '__main__':
    model, vector = train('data/train_cut.csv')
    saveModel(model, vector)

# SVM all_data
# 0.22464919346662812 0.6332072460186683 0.2733152651004384

# SVM cut_data
# 0.36485795579018365 0.761904423362893 0.4237380707313948

#LogisticRegression cut_data
# 0.30260595967253046 0.6963076359223289 0.36191005895990425

# 决策树 cut_data
# 0.39966075264748163 0.6702155265494737 0.4376566012394919

# 随机森林 cut_data 10
# 0.37513547430195204 0.6725904989301547 0.4164320671775545

#随机森林 cut_data 100
# 0.38108857875525876 0.756439861029364 0.4244387384325999

# AdaBoost cut_data
# 0.20123084162052104 0.17191289106479585 0.17723527339604014

# GradientBoost
# 0.3147953116183219 0.7133429668955028 0.3755094982860324
