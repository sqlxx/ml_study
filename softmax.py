import torch
import torchvision
from torch.utils import data
from torchvision import transforms

if __name__ == "__main__":
    # 通过ToTensor类将PIL图片或者numpy.ndarray数据转换成形状为(C,H,W)的Tensor格式，并且归一化到[0.0,1.0]之间
    trans = transforms.ToTensor()

    mnist_train = torchvision.datasets.FashionMNIST( root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST( root="./data", train =False, transform=trans, download=True)   

    print(len(mnist_train), len(mnist_test))  # 60000 10000
