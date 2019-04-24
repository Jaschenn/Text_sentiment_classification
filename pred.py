import keras
import numpy as np
import jieba
import pandas as pd
model = keras.models.load_model("./model/model.h5")
gooddata = list(jieba.cut("热水器已安装上了性价比不错从发货到安装一切顺利 卖家态度很好，物流快 顺丰不是盖的   安装师傅态度和服务也不错   安装材料花了98元"
                          "不知道是不是首次用的原因 不过保温效果不错 今天还很热   很不错的网购   等用过几次再来追评 "
                          "  热水器已安装上了性价比不错从发货到安装一切顺利 卖家态度很好，物流快 顺丰不是盖的   安装师傅态度和服务也不错   安装材料花了98元"
                          "不知道是不是首次用的原因 不过保温效果不错 今天还很热   很不错的网购   等用过几次再来追评    "))


baddata = list(jieba.cut("人多，昏暗，难闻，不好玩。是我见过最差的海洋馆，据说建设年代久远了。里面奶瓶喂鱼25元，小孩小孩。"))


min_count = 1
max_len = 100

def doc2num(s, maxlen):
    content = []
    for i in gooddata:
        content.extend(i)

    abc = pd.Series(content).value_counts()
    abc = abc[abc >= min_count]
    abc[:] = list(range(1, len(abc) + 1))
    abc[''] = 0  # 添加空字符串用来补
    word_set = set(abc.index)
    s = [i for i in s if i in word_set]
    s = s[:maxlen] + ['']*max(0, maxlen-len(s))
    return list(abc[s])

gooddata = doc2num(gooddata, max_len)
gooddata = np.reshape(gooddata, (1, 100))
baddata = doc2num(baddata, max_len)
baddata = np.reshape(baddata, (1, 100))
print(model.predict_classes(gooddata))
print(model.predict_classes(baddata))
