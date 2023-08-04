from sklearn.tree import DecisionTreeClassifier
from DataUtils import getTokens, modelfile_path, vectorfile_path
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def getDataFromFile(filename):
    with open(filename, "r", encoding="utf8") as f:
        inputurls, y = [], []
        for line in f:
            link = line.strip()
            inputurls.append(link)

    print("read ok!")
    return inputurls

def loadModel():
    file1 = modelfile_path
    with open(file1, 'rb') as f1:
        model = pickle.load(f1)
    f1.close()

    file2 = vectorfile_path
    with open(file2, 'rb') as f2:
        vector = pickle.load(f2)
    f2.close()
    return model, vector

model, vector = loadModel()
all_urls = getDataFromFile("./data/test(unlabeled).csv")
x = vector.transform(all_urls)
y_predict = model.predict(x)

print(len(all_urls))
print(len(y_predict))

with open("./data/test(labeled).csv", "w", encoding="utf8") as w:
    for link, label in zip(all_urls, y_predict):
        w.write(link+","+label+"\n")