from matplotlib.pyplot import axis
import numpy as np
from .autograd import Tensor
import struct
import gzip
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Transform:

    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            return img[:, ::-1, :]
        else:
            return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):

    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding,
                                             high=self.padding + 1,
                                             size=2)
        ### BEGIN YOUR SOLUTION
        h, w, c = img.shape
        pad = np.zeros((h + 2 * self.padding, w + 2 * self.padding, c))
        pad[self.padding:self.padding + h,
            self.padding:self.padding + w, :] = img
        x = self.padding + shift_x
        y = self.padding + shift_y
        crop = pad[x:x + h, y:y + w, :]
        return crop
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(
                np.arange(len(dataset)),
                range(batch_size, len(dataset), batch_size))
        else:
            arr = np.arange(len(dataset))
            np.random.shuffle(arr)
            self.ordering = np.array_split(
                arr, range(batch_size, len(dataset), batch_size))
        self.idx = -1

    def __iter__(self):
        ### BEGIN YOUR SOLUTION ###

        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        self.idx += 1
        if self.idx >= len(self.ordering):
            raise StopIteration()
        samples = self.dataset[self.ordering[self.idx]]
        # samples = list(zip(*samples))
        # ret = [np.concatenate([x]) for x in samples]
        return [Tensor(x) for x in samples]

        ### END YOUR SOLUTION


class MNISTDataset(Dataset):

    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        with gzip.open(image_filename) as file:
            magic, num, rows, cols = struct.unpack(">IIII", file.read(16))
            self.images = np.fromstring(file.read(),
                                        dtype=np.uint8).reshape(-1, 784)
        with gzip.open(label_filename) as file:
            magic, n = struct.unpack(">II", file.read(8))
            self.labels = np.fromstring(file.read(), dtype=np.uint8)

        self.images = np.float32(self.images) / 255.
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        if isinstance(index, (Iterable, slice)):
            img = [i.reshape((28, 28, 1)) for i in self.images[index]]
        # elif isinstance(index, slice):
        #     img = [i.reshape((28, 28, 1)) for i in self.images[index]]
        else:
            img = [self.images[index].reshape((28, 28, 1))]

        if self.transforms:
            for tsf in self.transforms:
                img = [tsf(x) for x in img]

        return [np.stack(img), self.labels[index]]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.images)
        ### END YOUR SOLUTION


class NDArrayDataset(Dataset):

    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
