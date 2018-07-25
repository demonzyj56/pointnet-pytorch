"""Dataset definition for Stanford indoor3d."""
import os
import logging
import h5py
import numpy as np
import torch
import torch.utils.data

logger = logging.getLogger(__name__)


def load_h5(filename):
    """Utility function for loading h5 files."""
    f = h5py.File(filename)
    return (f['data'][:], f['label'][:])


class Indoor3D(torch.utils.data.Dataset):
    """Stanford Indoor3D dataset for 3D semantic segmentation."""

    def __init__(self, root, train=True, test_area=6, transform=None,
                 num_points=1024):
        """
        Parameters
        ----------
        root: string
            Root path of the indoor3d data.
        train: bool
            Whether should load training set or test set.
        test_area: int, one of {1, 2, 3, 4, 5, 6}
            Which area (1-6) in the dataset should be regarded as test set.
        transform: callable
            Specify transform (augmentation) for point clouds before output.
        num_points: int
            Number of data points for each point cloud. This selects the first
            num_points for each point cloud.
        """
        data, label = [], []
        for i in range(24):
            fname = os.path.join(root, 'ply_data_all_{:d}.h5'.format(i))
            assert os.path.exists(fname)
            d, l = load_h5(fname)
            data.append(d)
            label.append(l)
        data = np.concatenate(data, axis=0)
        label = np.concatenate(label, axis=0)
        self.num_classes = len(np.unique(label))
        room_list = [l.strip() for l in
                     open(os.path.join(root, 'room_filelist.txt'), 'r')]
        assert len(room_list) == data.shape[0]
        assert len(room_list) == label.shape[0]
        area = 'Area_{:d}'.format(test_area)
        train_idx = [i for i, room in enumerate(room_list) if not room.starts_with(area)]
        test_idx = [i for i, room in enumerate(room_list) if room.starts_with(area)]
        if train:
            self.data = data[train_idx]
            self.label = label[train_idx]
        else:
            self.data = data[test_idx]
            self.label = label[test_idx]
        self.num_points = min(num_points, self.data.shape[1])
        self.data = self.data[:, :self.num_points, :].permute(0, 2, 1).astype(np.float32)
        self.label = self.label[:, :self.num_points].astype(np.int32)
        self.train = train
        self.transform = transform

    def __len__(self):
        """Number of samples in the dataset."""
        return self.data.shape[0]

    def __getitem__(self, index):
        """Get the i-th point cloud."""
        data = self.data[index]
        if self.transform is not None:
            data = self.transform(data)  # 13 x num_points
        label = self.label[index]  # num_points
        return torch.from_numpy(data), torch.LongTensor(label)


def collate_fn(batch):
    """Collate function for Indoor3D."""
    data = [b[0] for b in batch]
    label = [b[1] for b in batch]
    return torch.stack(data, dim=0), torch.stack(label, dim=0)
