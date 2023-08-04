"""
拼接训练集 temp train1 train1补充 train2 train3 ===> train
"""

import pandas as pd

# train1      0, 1, 2, 5, 6, 9, 10, 11
# train1补充   3, 4, 7, 8,
# temp        1, 2
# train2      0, 2, 10, 11
# train3      1, 2

# train23 缺少： 3，4，5，6，7，8，9，12

id2label = {0: "正常网址", 1: "购物消费", 2: "婚恋交友", 3: "假冒身份",
            4: "钓鱼网站", 5: "冒充公检法", 6: "平台诈骗", 7: "招聘兼职",
            8: "杀猪盘", 9: "博彩赌博", 10: "信贷理财", 11: "刷单诈骗",
            12: "中奖诈骗"}
label2id = dict([(v, k) for k, v in id2label.items()])

X = []
Y = []

length = 256
i = 0
with open("./train_link_text_label_{}.csv".format(length), "w", encoding="utf8") as w:
    # with open("./train1.csv", "r", encoding="utf8") as f:
    #     for line in f:
    #         i += 1
    #         link, label = line.split(",")
    #         label = id2label[int(label.strip())]
    #
    #         w.write(link + "," + label + "\n")
    # print(i)
    # with open("./train1补充.csv", "r", encoding="utf8") as f:
    #     for line in f:
    #         i += 1
    #         link, label = line.split(",")
    #         label = id2label[int(label.strip())]
    #
    #         w.write(link + "," + label + "\n")
    # print(i)
    with open("./temp.csv", "r", encoding="utf8") as f:
        for line in f:
            i += 1
            link, text, label = line.split(",")
            label = id2label[int(label.strip())]
            text = text.replace(",", "")

            w.write(link + "," + text[:length] + "," + label + "\n")
    with open("./train2.csv", "r", encoding="utf8") as f:
        for line in f:
            i += 1
            link, text, label = line.split(",")
            label = label.strip()
            text = text.replace(",","")

            w.write(link + "," + text[:length] + "," + label + "\n")

    print(i)
    with open("./train3.csv", "r", encoding="utf8") as f:
        for line in f:
            i += 1
            link, text, label = line.split(",")

            label = id2label[int(label.strip())]
            text = text.replace(",", "")

            # w.write(link + "," + label + "\n")
            w.write(link + "," + text[:length] + "," + label + "\n")
print(i)
