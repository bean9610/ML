from gensim.models import word2vec

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

class MySentences(object):
    def __init__(self,fname):
        self.fname=fname

    def __iter__(self):
        for line in open(self.fname,"r"):
            yield line.split()

def w2vTrain(f_input,mode_ouput):
    sentences=MySentences(DataDir+f_input)
    w2v_model=word2vec.Word2Vec(sentences,min_count=MIN_COUNT,workers=CPU_NUM,size=VEC_SIZE,window=CONTEXT_WINDOW)
    w2v_model.save(ModelDir+model_output)

DataDir="./"
ModelDir="./w2v_files/"
MIN_COUNT=4  #词频小于4删除
CPU_NUM=1  #cpu并行数
VEC_SIZE=20 #特征向量维度
CONTEXT_WINDOW=5 #上下文词语

f_input = "bioCorpus_5000.txt"
model_output = "test_w2v_model"

w2vTrain(f_input, model_output)

w2v_model = word2vec.Word2Vec.load(ModelDir+model_output)

print(w2v_model.most_similar('body')) # 结果一般

w2v_model.most_similar('heart') # 结果太差

from nltk.corpus import stopwords

StopWords = stopwords.words('english')

StopWords[:20]

# 重新训练
# 模型训练函数
def w2vTrain_removeStopWords(f_input, model_output):
    sentences = list(MySentences(DataDir+f_input))
    for idx,sentence in enumerate(sentences):
        sentence = [w for w in sentence if w not in StopWords]
        sentences[idx]=sentence
    w2v_model = word2vec.Word2Vec(sentences, min_count = MIN_COUNT,
                                  workers = CPU_NUM, size = VEC_SIZE)
    w2v_model.save(ModelDir+model_output)

w2vTrain_removeStopWords(f_input, model_output)
w2v_model = word2vec.Word2Vec.load(ModelDir+model_output)

print(w2v_model.most_similar('heart'))# 结果一般