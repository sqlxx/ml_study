import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt

def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):

    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()

    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        if titles:
            ax.set_title(titles[i])
    return axes

def load_data_fashion_mnist(batch_size, resize=None):
    # 组件transform流水线
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))

    trans = transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST( root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST( root="./data", train =False, transform=trans, download=True)   

    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4))

if __name__ == "__main__":
    train_iter, test_iter = load_data_fashion_mnist(256)

    for X, y in train_iter:
        print(X.shape, X.dtype, y.shape, y.dtype)
        break


