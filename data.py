import torch
import numpy as np

class DataManager(object):
    def __init__(self, max_samples, x_dim, y_dim, remove_random=False):
        self.max_samples = max_samples
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.data_x = torch.zeros((max_samples, x_dim))
        self.data_y = torch.zeros((max_samples, y_dim))
        self.remove_random = remove_random

        self.cursor = 0  # where should the next data point be
        self.number_of_samples = 0  # num of samples currently in the buffer

        self.data_x_slice, self.data_y_slice = None, None

    def add(self, x, y):
        bs = x.size(0)

        if self.cursor + bs <= self.max_samples:
            self.data_x[self.cursor: self.cursor + bs, :] = x
            self.data_y[self.cursor: self.cursor + bs, :] = y
        else:
            import ipdb; ipdb.set_trace(context=5)
            self.data_x[self.cursor:, :] = x[: self.max_samples - self.cursor]
            self.data_x[:bs - self.max_samples + self.cursor, :] = x[self.max_samples - self.cursor:]
            self.data_y[self.cursor:, :] = y[: self.max_samples - self.cursor]
            self.data_y[:bs - self.max_samples + self.cursor, :] = y[self.max_samples - self.cursor:]

        # update number of samples
        self.number_of_samples = min(self.number_of_samples + bs, self.max_samples)

        # move cursor
        if self.remove_random and self.number_of_samples == self.max_samples:
            self.cursor = np.random.randint(0, self.max_samples)
        else:
            self.cursor = (self.cursor + bs) % self.max_samples

    def sample(self, bs):
        idxs = np.arange(self.number_of_samples)
        np.random.shuffle(idxs)
        idxs = torch.as_tensor(idxs[:bs]).long()

        return self.data_x[idxs], self.data_y[idxs]

    def get_batch_without_replacement(self, bs):
        if self.data_x_slice is None and self.data_y_slice is None:
            idxs = np.arange(self.number_of_samples)
            np.random.shuffle(idxs)
            idxs = torch.as_tensor(idxs).long()
            self.data_x_slice = self.data_x[idxs]
            self.data_y_slice = self.data_y[idxs]

        batch_x, batch_y = self.data_x_slice[:bs], self.data_y_slice[:bs]

        if bs < self.data_x_slice.size(0):
            self.data_x_slice, self.data_y_slice = self.data_x_slice[bs:], self.data_y_slice[bs:]
            last_batch = False
        else:
            self.data_x_slice, self.data_y_slice = None, None
            last_batch = True

        return batch_x, batch_y, last_batch

