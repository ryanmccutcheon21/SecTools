from __future__ import annotations

try:
    import torch
    from torch.utils.data import DataLoader, Subset, TensorDataset
    import torchvision
    import torchvision.transforms as transforms
except Exception:
    torch = None
    DataLoader = None
    Subset = None
    TensorDataset = None
    torchvision = None
    transforms = None

def synthetic_loaders(n_train=5000, n_test=2000, batch_size=256, channels=1, img_size=28, n_classes=10):
    x_train = torch.randn(n_train, channels, img_size, img_size)
    y_train = torch.randint(0, n_classes, (n_train,))
    x_test = torch.randn(n_test, channels, img_size, img_size)
    y_test = torch.randint(0, n_classes, (n_test,))
    return (
        DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False),
    )

def load_mnist(batch_size=256, n_train=5000, n_test=2000):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    try:
        train_ds = torchvision.datasets.MNIST("./data", train=True, download=True, transform=tf)
        test_ds = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tf)
        train_ds = Subset(train_ds, range(min(n_train, len(train_ds))))
        test_ds = Subset(test_ds, range(min(n_test, len(test_ds))))
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        )
    except Exception:
        return synthetic_loaders(n_train=n_train, n_test=n_test, batch_size=batch_size, channels=1, img_size=28, n_classes=10)

def load_cifar10(batch_size=256, n_train=5000, n_test=2000):
    tf = transforms.Compose([transforms.ToTensor()])
    try:
        train_ds = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=tf)
        test_ds = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=tf)
        train_ds = Subset(train_ds, range(min(n_train, len(train_ds))))
        test_ds = Subset(test_ds, range(min(n_test, len(test_ds))))
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        )
    except Exception:
        return synthetic_loaders(n_train=n_train, n_test=n_test, batch_size=batch_size, channels=3, img_size=32, n_classes=10)

def load_dataset(name: str, batch_size=256, n_train=5000, n_test=2000):
    if name == "cifar10":
        return load_cifar10(batch_size=batch_size, n_train=n_train, n_test=n_test)
    return load_mnist(batch_size=batch_size, n_train=n_train, n_test=n_test)
