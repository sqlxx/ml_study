import torch
import random
from torch.utils import data
import matplotlib.pyplot as plot
from torch import nn

def synthetic_data(w, b, num_examples):
    """Generate y = Xw + b + noise."""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples )])
        yield features[batch_indices], labels[batch_indices]

def linreg(X, w, b):
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad/batch_size
            param.grad.zero_()

## 利用框架实现
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


if __name__ == "__main__":
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    print('features:', features[0], '\nlabel:', labels[0])

    # plot.scatter(features[:, 1].numpy(), labels.numpy(), 1)
    # plot.show()


    w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    lr = 0.03
    num_epochs = 3
    loss_func = squared_loss
    net = linreg

    batch_size = 10

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            loss = loss_func(net(X, w, b), y) 
            loss.sum().backward()
            sgd([w, b], lr, batch_size)

        with torch.no_grad():
            train_l = loss_func(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

    print(f'true_w: {true_w}, w: {w.reshape(true_w.shape)}')
    print(f'true_b: {true_b}, b: {b}')

    # 利用pytorch框架实现
    loss_func = nn.MSELoss()
    net = nn.Sequential(nn.Linear(2,1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)
    trainer = torch.optim.SGD(net.parameters(), lr=lr   )
    di = load_array((features, labels), batch_size=10)

    for epoch in range(num_epochs):
        for X, y in di:
            loss = loss_func(net(X), y)
            trainer.zero_grad()
            loss.backward()
            trainer.step()
        loss = loss_func(net(features), labels)
        print(f'epoch {epoch + 1}, loss { loss:f}')
    
    print(f'true_w: {true_w}, w: {net[0].weight.data.reshape(true_w.shape)}')
    print(f'true_b: {true_b}, b: {net[0].bias.data}')


    



