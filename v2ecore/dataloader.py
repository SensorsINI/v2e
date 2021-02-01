"""customized Pytorch dataloader

    @author: Zhe He
    @contact: zhehe@student.ethz.ch
    @latest update: 2019-May-27th
"""
import torch.utils.data as data

import glob
import numpy as np
from PIL import Image


class Frames(data.Dataset):

    """
        Load frames from an N-d array, and transform them into tensor.
        @Author:
            - Zhe He
            - zhehe@student.ethz.ch

        @Members:
            array: N-d numpy array.
            transform: Compose object.

        @Methods:
            __getitem__: List(Tensor, Tensor)
                return a pair of (frame0, frame1).
            __len__: int
                return the length of the dataset.
            __repr__: str
                return printable representation of the class.
    """

    def __init__(self, array, transform=None):

        """
            @Parameters:
                array: N-d numpy array.
                transform: Compose object.
        """

        self.array = array
        self.transform = transform
        self.origDim = array.shape[2], array.shape[1]
        self.dim = (int(self.origDim[0] / 32) * 32,
                    int(self.origDim[1] / 32) * 32)

    def __getitem__(self, index):

        """Return an item from the dataset.

            @Parameter:
                index: int.
            @Return: List(Tensor, Tensor).
        """

        sample = []
        # Loop over for all frames corresponding to the `index`.
        for image in [self.array[index], self.array[index + 1]]:
            # Open image using pil.
            image = Image.fromarray(image)
            image = image.resize(self.dim, Image.ANTIALIAS)
            # Apply transformation if specified.
            if self.transform is not None:
                image = self.transform(image)
            sample.append(image)
        return sample

    def __len__(self):

        """Return the size of the dataset.
            @Return: int.
        """
        # from SuperSloMo github: https://github.com/avinashpaliwal
        # /Super-SloMo/blob/a3be3dbc363f48f2cdaafc7224597dbd5a0d9370
        # /dataloader.py#L412
        #
        # Using `-1` so that dataloader accesses only upto
        # frames [N-1, N] and not [N, N+1] which because frame
        # N+1 doesn't exist.
        # ??? why is array 1 less than number of images?
        # Is it to avoid double processing the last frame of batch?
        return self.array.shape[0] - 1

    def __repr__(self):

        """Return printable representations of the class.
            @Return: str.
        """

        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp,
            self.transform.__repr__().replace('\n',
                                              '\n' + ' ' * len(tmp)))
        return fmt_str


class FramesDirectory(data.Dataset):

    """
        Load frames from a directory that has individual frame records,
        and transform them into tensor.
    """

    def __init__(self, folder_path, ori_dim, parsing="/*.npy", transform=None):

        """
            @Parameters:
                array: N-d numpy array.
                transform: Compose object.
        """

        self.files = sorted(glob.glob("{}".format(folder_path)+parsing))

        self.transform = transform
        self.origDim = ori_dim
        #  self.origDim = array.shape[2], array.shape[1]
        self.dim = (int(self.origDim[0] / 32) * 32,
                    int(self.origDim[1] / 32) * 32)

    def __getitem__(self, index):

        """Return an item from the dataset.

            @Parameter:
                index: int.
            @Return: List(Tensor, Tensor).
        """

        sample = []

        image_1 = np.load(self.files[index])
        image_2 = np.load(self.files[index+1])
        # Loop over for all frames corresponding to the `index`.
        for image in [image_1, image_2]:
            # Open image using pil.
            image = Image.fromarray(image)
            image = image.resize(self.dim, Image.ANTIALIAS)
            # Apply transformation if specified.
            if self.transform is not None:
                image = self.transform(image)
            sample.append(image)
        return sample

    def __len__(self):

        """Return the size of the dataset.
            @Return: int.
        """
        # from SuperSloMo github: https://github.com/avinashpaliwal
        # /Super-SloMo/blob/a3be3dbc363f48f2cdaafc7224597dbd5a0d9370
        # /dataloader.py#L412
        #
        # Using `-1` so that dataloader accesses only upto
        # frames [N-1, N] and not [N, N+1] which because frame
        # N+1 doesn't exist.
        # ??? why is array 1 less than number of images?
        # Is it to avoid double processing the last frame of batch?
        # No, because we need to read a pair of image
        return len(self.files)-1

    def __repr__(self):

        """Return printable representations of the class.
            @Return: str.
        """

        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp,
            self.transform.__repr__().replace('\n',
                                              '\n' + ' ' * len(tmp)))
        return fmt_str
