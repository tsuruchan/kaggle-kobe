import scipy as sp
import pandas as pd
import argparse
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList, FunctionSet
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
from pandas import DataFrame as DF
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

def test_p():
    import math
    ans = []
    pre = []
    for i in range(5000):
        temp = predictor(xp.asarray([[test_X[i]]]))
        pro0, pro1 = temp.data[0][0], temp.data[0][1]
        one = math.exp(pro1) / (math.exp(pro0) + math.exp(pro1))
        zero = math.exp(pro0) / (math.exp(pro0) + math.exp(pro1))
        pre.append((zero, one))
        if test_y[i] == 0:
            ans.append((1, 0))
        elif test_y[i] == 1:
            ans.append((0, 1))
        else:
            print("error")
    
    return logloss(ans, pre)
        
        

# GPU設定
parser = argparse.ArgumentParser(description='Chainer example: CIFAR-10')
parser.add_argument('--gpu', '-gpu', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')


# GPUが使えるか確認
args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

X = pd.read_csv("kobe_train.csv", header = None)
y = pd.read_csv("kobe_result.csv", header = None)
predict = pd.read_csv("kobe_pred.csv", header = None)
X = np.array(X)
y = np.array(y)
predict = np.array(predict)
X = X.astype(np.float32)
y = y.astype(np.int32)
predict = predict.astype(np.float32)

train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=20000/25697, random_state=0)

# parameter
SIZE = 256
r = 0.2
n_epoch = 500
N = train_X.shape[0]
N_test = test_X.shape[0]
batchsize = 100


model = chainer.FunctionSet(l1 = L.Linear(125, SIZE),
                            l2 = L.Linear(SIZE,SIZE),
                            l3 = L.Linear(SIZE, SIZE),
                            l4 = L.Linear(SIZE, SIZE),
                            l5 = L.Linear(SIZE, 2))

# GPU使用のときはGPUにモデルを転送
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# 順伝播

def forward(X_data, y_data, train=True):
    x, t = chainer.Variable(X_data), chainer.Variable(y_data)
    h = F.dropout(model.l1(x), ratio=r, train=train)
    h = F.dropout(model.l2(h), ratio=r, train=train)
    h = F.dropout(model.l3(h), ratio=r, train=train)
    h = F.dropout(model.l4(h), ratio=r, train=train)
    y = model.l5(h)

    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

def predictor(X_data, train=False):
    x = chainer.Variable(X_data)
    h = F.dropout(model.l1(x), ratio=r, train=train)
    h = F.dropout(model.l2(h), ratio=r, train=train)
    h = F.dropout(model.l3(h), ratio=r, train=train)
    h = F.dropout(model.l4(h), ratio=r, train=train)
    y = model.l5(h)

    return y



# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

# 値格納配列
l = []
a = []

# Learning loop
for epoch in range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in range(0, N, batchsize):
        X_batch = xp.asarray(train_X[perm[i:i + batchsize]])
        y_batch = xp.asarray(train_y[perm[i:i + batchsize]])

        batxh_len = X_batch.shape[0]
        y_batch = y_batch.reshape(y_batch.shape[0], )

        # 勾配を初期化
        optimizer.zero_grads()
        # 順伝播させて誤差と精度を計算
        loss, acc = forward(X_batch, y_batch)
        # バックプロパゲーション
        loss.backward()
        # 最適化ルーティーンを実行
        optimizer.update()

        sum_loss += float(loss.data) * batxh_len
        sum_accuracy += float(acc.data) * batxh_len
    
    print('train mean loss={}, accuracy={}'.format(sum_loss / N, sum_accuracy / N))
    

    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in range(0, N_test, batchsize):
        X_batch = xp.asarray(test_X[i:i + batchsize])
        y_batch = xp.asarray(test_y[i:i + batchsize])

        batxh_len = X_batch.shape[0]

        y_batch = y_batch.reshape(y_batch.shape[0], )
        # 順伝播させて誤差と精度を計算
        loss, acc = forward(X_batch, y_batch, train=False)

        sum_loss += float(loss.data) * batxh_len
        sum_accuracy += float(acc.data) * batxh_len

    sl, sa = sum_loss / N_test, sum_accuracy / N_test
    a.append(sa)
    l.append(sl)
    print('test  mean loss={}, accuracy={}'.format(sl, sa))
    

def p(data):
    import math
    ans = []
    for i in range(len(data)):
        temp = predictor(xp.asarray([[data[i]]]))
        pro0, pro1 = temp.data[0][0], temp.data[0][1]
        ans.append(math.exp(pro1) / (math.exp(pro0) + math.exp(pro1)))

    return ans
