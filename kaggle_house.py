import downloader
import pandas as pd
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from device import device

downloader.DATA_HUB['kaggle_house_train'] = (downloader.DATA_URL + 'kaggle_house_pred_train.csv', '585e9cc93e70b39160e7921475f9bcd7d31219ce')
downloader.DATA_HUB['kaggle_house_test'] = (downloader.DATA_URL + 'kaggle_house_pred_test.csv', 'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

train_data = pd.read_csv(downloader.download('kaggle_house_train'))
test_data = pd.read_csv(downloader.download('kaggle_house_test'))

print(train_data.shape)
print(test_data.shape)

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

numeric_featues = all_features.dtypes[all_features.dtypes != 'object'].index

all_features[numeric_featues] = all_features[numeric_featues].apply(lambda x: (x - x.mean()) / (x.std()))

all_features[numeric_featues] = all_features[numeric_featues].fillna(0)
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features = all_features.astype(np.float32)

n_train = train_data.shape[0]
print('training data is', n_train)
train_features = torch.tensor(all_features[:n_train].to_numpy(), dtype=torch.float32, device=device)
test_features = torch.tensor(all_features[n_train:].to_numpy(), dtype=torch.float32, device=device)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32, device=device)

loss = nn.MSELoss()
in_features = train_features.shape[1]
print(train_features.shape)

def get_net():
  net = nn.Sequential(nn.Linear(in_features, 1))
  return net.to(device)

def log_rmse(net, features, labels):
  clipped_preds = torch.clamp(net(features), 1, float('inf'))
  rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
  return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay, batch_size):
  train_ls, test_ls = [], []
  train_iter = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_features, train_labels), batch_size, shuffle=True)

  optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate, weight_decay=weight_decay)

  for epoch in range(num_epochs):
    for X, y in train_iter:
      optimizer.zero_grad()
      vloss = loss(net(X), y)
      vloss.backward()
      optimizer.step()
    train_ls.append(log_rmse(net, train_features, train_labels))
    if test_labels is not None:
      test_ls.append(log_rmse(net, test_features, test_labels))

  return train_ls, test_ls

def get_k_fold_data(k, i, X, y): # k折交叉验证， 第i折作为验证数据，其余作为训练数据
  assert k > 1
  fold_size = X.shape[0] // k
  X_train, y_train = None, None
  for j in range(k):
    idx = slice(j * fold_size, (j + 1) * fold_size)
    X_part, y_part = X[idx, :], y[idx]
    if j == i:
      X_valid, y_valid = X_part, y_part
    elif X_train is None:
      X_train, y_train = X_part, y_part
    else:
      X_train = torch.cat([X_train, X_part], 0)
      y_train = torch.cat([y_train, y_part], 0)
  return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
  train_l_sum, valid_l_sum = 0, 0
  for i in range(k):
    data = get_k_fold_data(k, i, X_train, y_train)
    net = get_net()
    train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
    train_l_sum += train_ls[-1]
    valid_l_sum += valid_ls[-1]

    if i == 0:
      epochs = range(1, num_epochs + 1)

      plt.plot(epochs, train_ls, label='train')
      plt.plot(epochs, valid_ls, label='valid')

      plt.xlabel('epoch')
      plt.ylabel('rmse')
      plt.yscale('log')
      plt.legend()
      plt.show()

    print('fold %d, train rmse %f, valid rmse %f' % (i + 1, train_ls[-1], valid_ls[-1]))
  
  return train_l_sum / k, valid_l_sum / k


k, num_epochs, lr, weight_decay, batch_size = 5, 120, 4, 0.01, 16
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (k, train_l, valid_l))

def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
  net = get_net()
  train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
  plt.plot(range(1, num_epochs+1), train_ls )
  plt.xlabel('epoch')
  plt.ylabel('rmse')
  plt.yscale('log')
  plt.show()

  print(f'training rmse: {float(train_ls[-1]):f}')
  preds = net(test_features).cpu().detach().numpy()
  test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
  submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
  submission.to_csv('submission.csv', index=False)


# train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)
