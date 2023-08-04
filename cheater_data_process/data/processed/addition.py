id2label = {0:"正常网址",1:"购物消费",2:"婚恋交友",3:"假冒身份",4:"钓鱼网站",5:"冒充公检法",6:"平台诈骗",
            7:"招聘兼职",8:"杀猪盘",9:"博彩赌博",10:"信贷理财",11:"刷单诈骗",12:"中奖诈骗"}
label2id = dict([(v,k) for k,v in id2label.items()])

labels = ["假冒身份","钓鱼网站","冒充公检法","平台诈骗","招聘兼职","杀猪盘","博彩赌博","中奖诈骗"]
label2num = dict([(l,0) for l in labels])

# {'假冒身份': 173, '钓鱼网站': 4658, '冒充公检法': 2, '平台诈骗': 2603, '招聘兼职': 11, '杀猪盘': 883, '博彩赌博': 595, '中奖诈骗': 7}
with open("./addition.csv", "w", encoding="utf8") as w:
    with open("./train1.csv", "r", encoding="utf8") as f:
        for line in f:
            link, label = line.split(",")
            label = id2label[int(label.strip())]
            text = link
            if(label in labels):
                label2num[label] += 1

                w.write(text+","+label+"\n")

    with open("./train1补充.csv", "r", encoding="utf8") as f:
        for line in f:
            link, label = line.split(",")
            label = id2label[int(label.strip())]
            text = link
            if (label in labels):
                label2num[label] += 1

                w.write(text+","+label+"\n")
print(label2num)