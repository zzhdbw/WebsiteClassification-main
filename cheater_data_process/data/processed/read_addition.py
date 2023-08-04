import pandas as pd

def read_addition(num):
    data_list = []
    with open("./addition_spyder_{}.csv".format(num), "r", encoding="utf8") as r:
        for line in r:
            link, text, label = line.strip().split(",")
            if(not text or text=="None"):
                continue
            data_list.append([link, text, label])
    return data_list

if __name__ == '__main__':
    all_data_list = []
    for i in range(500, 8001, 500):
        data_list = read_addition(i)
        all_data_list.extend(data_list)
    data_list = read_addition(8932)
    all_data_list.extend(data_list)
    print(len(all_data_list))

    with open("./addition_spyder.csv", "w", encoding="utf8") as w:
        for link, text, label in all_data_list:
            w.write(link+","+text+","+label+"\n")
