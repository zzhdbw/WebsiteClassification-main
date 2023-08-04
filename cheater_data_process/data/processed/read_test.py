"""将爬取到的文本写入到test集"""

from collections import Counter

def read_test(num):
    nosense_list = ["None",
                    "页面未找到",
                    "温馨提示网站暂时无法",
                    "您好尊敬的网易用户"]

    text_list = []
    with open("test(unlabeled)_spyder_{}.csv".format(num), "r", encoding="utf8") as f:
        for line in f:
            link, text = line.strip().split(",")
            if (not text):
                continue
            is_nosense = False
            for nosense in nosense_list:
                if (nosense in text):
                    is_nosense = True
                    break
            if (is_nosense):
                continue

            text_list.append([link, text])
    return text_list

all_text_list = []
for i in range(100000, 1000001, 100000):
    text_list = read_test(i)
    all_text_list.extend(text_list)

link_text_dict = dict()
for link, text in all_text_list:
    link_text_dict[link.strip()] = text

with open("test(unlabeled).csv", "r", encoding="utf8") as r:
    with open("test(unlabeled)_spyder.csv", "w", encoding="utf8") as w:
        for link in r:
            link = link.strip()
            if(link in link_text_dict.keys()):
                text = link_text_dict[link]
                w.write(link + "," + text +"\n")
            else:
                w.write(link + "," + "\n")
