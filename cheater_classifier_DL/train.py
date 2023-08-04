import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import Module, Embedding, Linear, CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertTokenizer
from modules import Model_BERT, Model_BERT_BILSTM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_len = 256
word_dim = 300
batch_size = 4
learning_rate = 2e-6
epoch = 40

id2label = {0: "正常网址", 1: "购物消费", 2: "婚恋交友", 3: "假冒身份", 4: "钓鱼网站", 5: "冒充公检法", 6: "平台诈骗",
            7: "招聘兼职", 8: "杀猪盘", 9: "博彩赌博", 10: "信贷理财", 11: "刷单诈骗", 12: "中奖诈骗"}
label2id = dict([(v, k) for k, v in id2label.items()])

X = []
Y = []
label_set = set()

with open("data/train_link_text_label_256.csv", "r", encoding="utf8") as f:
    for line in f:
        link, text, label = line.strip().split(",")

        X.append(link+"#"+text.strip())
        Y.append(label.strip())

with open("data/addition_spyder.csv", "r", encoding="utf8") as f:
    for line in f:
        link, text, label = line.strip().split(",")

        X.append(link+"#"+text.strip())
        Y.append(label.strip())
print("read success")
print(len(X))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

class MyDataset(Dataset):
    def __init__(self, X, Y):
        super(MyDataset, self).__init__()
        self.X = X  # X
        self.Y = Y  # Y

    def __getitem__(self, item):
        return self.X[item], label2id[self.Y[item]]

    def __len__(self):
        return len(self.Y)

tokenizer = BertTokenizer.from_pretrained("./pretrained_model/bert-base-chinese")

def collate_fn(batch):
    batch_text = []
    batch_label = []

    for text, label in batch:
        text = tokenizer.encode(text,
                                padding="max_length",
                                max_length=max_len,
                                pad_to_max_length=True,
                                truncation=True)

        batch_text.append(text)
        batch_label.append(label)

    batch_text = torch.LongTensor(batch_text)
    batch_label = torch.LongTensor(batch_label)

    return batch_text.to(device), batch_label.to(device)

trainDataset = MyDataset(X_train, Y_train)
testDataset = MyDataset(X_test, Y_test)

trainDataLoader = DataLoader(trainDataset,
                             batch_size=batch_size,
                             shuffle=True,
                             collate_fn=collate_fn,
                             num_workers=0)

testDataLoader = DataLoader(testDataset,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=collate_fn,
                            num_workers=0)

myModel = Model_BERT_BILSTM().to(device)

loss_fn = CrossEntropyLoss()  # 损失函数
optimizer = Adam(lr=learning_rate, params=myModel.parameters())  # 优化器

best_f1 = 0.0
step = 0
for epo in range(epoch):
    lossz = 0
    myModel.train()
    for batch_text, batch_label in tqdm(trainDataLoader):
        myModel.train()
        optimizer.zero_grad()  # 梯度清零
        output = myModel(batch_text)  # 前向预测
        loss = loss_fn(output, batch_label)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 梯度下降
        lossz += loss

        step += 1
        if(step%1000 == 0):
            myModel.eval()
            predict_list = []
            true_list = []
            for batch_text, batch_label in testDataLoader:
                output = myModel(batch_text)
                output = torch.argmax(output, dim=-1)

                predict_list.extend(output)
                true_list.extend(batch_label)

            predict_list = [i.to("cpu").item() for i in predict_list]
            true_list = [i.to("cpu").item() for i in true_list]

            acc = accuracy_score(true_list, predict_list)
            p = precision_score(true_list, predict_list, average='macro')
            r = recall_score(true_list, predict_list, average='macro')
            f = f1_score(true_list, predict_list, average='macro')

            if (f >= best_f1):
                torch.save(myModel.state_dict(), 'ckpt/best_model_{}.pth'.format(str(f)))
                best_f1 = f

            print("epoch:{}, acc:{:.5}, P:{:.5}, R:{:.5}, F:{:.5}, best_F:{}".format(epo, acc, p, r, f, best_f1))

#bert
#epoch:26, acc:0.94314, P:0.85789, R:0.75764, F:0.79736, best_F:0.7973620920536753

#bert bilstm
# P:0.88429, R:0.86224, F:0.86842, best_F:0.8684237786898426