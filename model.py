model_3 = chainer.FunctionSet(l1=L.Linear(125, SIZE),
                              l2=L.Linear(SIZE, SIZE),
                              l3=L.Linear(SIZE, 2))

model_4 = chainer.FunctionSet(l1=L.Linear(125, SIZE),
                              l2=L.Linear(SIZE, SIZE),
                              l3=L.Linear(SIZE, SIZE),
                              l4=L.Linear(SIZE, 2))

model_5 = chainer.FunctionSet(l1=L.Linear(125, SIZE),
                              l2=L.Linear(SIZE, SIZE),
                              l3=L.Linear(SIZE, SIZE),
                              l4=L.Linear(SIZE, SIZE),
                              l5=L.Linear(SIZE, 2))

# nothing ativation function
def forward(X_data, y_data, train=True):
    x, t = chainer.Variable(X_data), chainer.Variable(y_data)
    h = F.dropout(model_5.l1(x), ratio=r, train=train)
    h = F.dropout(model_5.l2(h), ratio=r, train=train)
    h = F.dropout(model_5.l3(h), ratio=r, train=train)
    h = F.dropout(model_5.l4(h), ratio=r, train=train)
    y = model.l5(h)

    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

# relu activation function
def forward(X_data, y_data, train=True):
    x, t = chainer.Variable(X_data), chainer.Variable(y_data)
    h = F.dropout(F.relu(model_5.l1(x), ratio=r, train=train))
    h = F.dropout(F.relu(model_5.l2(x), ratio=r, train=train))
    h = F.dropout(F.relu(model_5.l3(x), ratio=r, train=train))
    h = F.dropout(F.relu(model_5.l4(x), ratio=r, train=train))
    y = model.l5(h)

    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

# tanh activation function
def forward(X_data, y_data, train=True):
    x, t = chainer.Variable(X_data), chainer.Variable(y_data)
    h = F.dropout(F.tanh(model_5.l1(x), ratio=r, train=train))
    h = F.dropout(F.tanh(model_5.l2(x), ratio=r, train=train))
    h = F.dropout(F.tanh(model_5.l3(x), ratio=r, train=train))
    h = F.dropout(F.tanh(model_5.l4(x), ratio=r, train=train))
    y = model.l5(h)

    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

