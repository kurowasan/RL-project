import numpy as np
import torch


class TileCoding(torch.nn.Module):
    def __init__(self, num_bins, num_tilings, limits):
        super(TileCoding, self).__init__()
        self.num_bins = num_bins
        self.num_tilings = num_tilings
        self.limits = limits  # (dim, 2) for each dim, lower and upper value

        self.num_dims = limits.shape[0]

        weights_shape = tuple([num_tilings] + [num_bins] * self.num_dims)
        self.weights = torch.nn.Parameter(torch.zeros(weights_shape))

        self.bins = np.zeros((self.num_tilings, self.num_dims, self.num_bins + 1))
        for tiling in range(self.num_tilings):
            for dim in range(self.num_dims):
                dim_range = self.limits[dim, 1] - self.limits[dim, 0]
                bin_size = dim_range / (self.num_bins + ( 1. / self.num_tilings - 1.))
                tiling_range = dim_range + bin_size * (1. - 1. / self.num_tilings)
                tiling_low = self.limits[dim, 0] - bin_size * tiling / self.num_tilings
                tiling_high = tiling_low + tiling_range
                self.bins[tiling, dim, :] = np.linspace(tiling_low, tiling_high, num=self.num_bins + 1)

    def x_dot_vector(self, state, vector):
        for tiling in range(self.num_tilings):
            bin_idx = self.get_bin_idx(tiling, state)
            if tiling == 0:
                element_sum = vector[tuple([tiling] + bin_idx)]
            else:
                element_sum = element_sum + vector[tuple([tiling] + bin_idx)]

        return element_sum

    def get_feature(self, state):
        x = torch.zeros_like(self.weights)

        for tiling in range(self.num_tilings):
            bin_idx = self.get_bin_idx(tiling, state)
            x[torch.as_tensor(tuple([tiling] + bin_idx)).long()] = 1.

        return x

    def forward(self, state):
        """Prediction function"""
        assert state.shape[0] == self.num_dims

        return self.x_dot_vector(state, self.weights)

    def get_bin_idx(self, tiling, state):
        bin_idx = []
        for dim in range(self.num_dims):
            if not (self.limits[dim, 0] <= state[dim] <= self.limits[dim, 1]): import ipdb; ipdb.set_trace(context=5)
            assert self.limits[dim, 0] <= state[dim] <= self.limits[dim, 1]
            bin_idx.append(np.digitize(state[dim], self.bins[tiling, dim]) - 1)

            if 0 > bin_idx[-1]:
                bin_idx[-1] = 0
            elif bin_idx[-1] > self.num_bins - 1:
                bin_idx[-1] = self.num_bins - 1

        return bin_idx


class FastTileCoding(torch.nn.Module):
    def __init__(self, num_bins, num_tilings, limits):
        super(FastTileCoding, self).__init__()
        self.num_bins = num_bins
        self.num_tilings = num_tilings
        self.limits = limits  # (dim, 2) for each dim, lower and upper value

        self.num_dims = limits.shape[0]

        weights_shape = tuple([num_tilings] + [num_bins] * self.num_dims)
        self.weights = torch.nn.Parameter(torch.zeros(weights_shape))

        self.bins = np.zeros((self.num_tilings, self.num_dims, self.num_bins + 1))
        for tiling in range(self.num_tilings):
            for dim in range(self.num_dims):
                dim_range = self.limits[dim, 1] - self.limits[dim, 0]
                bin_size = dim_range / (self.num_bins + ( 1. / self.num_tilings - 1.))
                tiling_range = dim_range + bin_size * (1. - 1. / self.num_tilings)
                tiling_low = self.limits[dim, 0] - bin_size * tiling / self.num_tilings
                tiling_high = tiling_low + tiling_range
                self.bins[tiling, dim, :] = np.linspace(tiling_low, tiling_high, num=self.num_bins + 1)
        self.bins = torch.Tensor(self.bins).float()
        self.bin_size = bin_size

    def forward(self, state):
        """Prediction function"""
        #import ipdb; ipdb.set_trace(context=5)
        assert state.size(1) == self.num_dims
        bs = state.size(0)
        # state (bs, num_dims) -> (bs, 1, num_dims, 1)
        state = state.unsqueeze(1).unsqueeze(3)
        # bins (num_tilings, num_dims, num_bins + 1)

        masks = (self.bins[:, :, :-1] <= state) & (state < self.bins[:, :, 1:])  # (bs, num_tilings, num_dims, num_bins)
        masks[:, :, :, -1] = masks[:, :, :, -1] | (state.squeeze(3) == self.bins[:, :, -1])  # (bs, num_tilings, num_dims, 1)
        masks = list(torch.unbind(masks, 2))  # list of num_dims masks of shape (bs, num_tilings, num_bins)
        mask_sum = 0
        for i, mask in enumerate(masks):
            shape = [bs, self.num_tilings] + [1] * self.num_dims
            shape[2 + i] = self.num_bins
            mask_sum = mask_sum + mask.view(tuple(shape)).long()  # (bs, num_tilings, 1, ..., 1, self.num_bins, 1, ..., 1)
        #print(mask_sum.shape)
        assert mask_sum.max() == self.num_dims
        final_mask = mask_sum == self.num_dims
        # Make sure exactly one weight per tiling is activated
        if (1 != final_mask.sum(dim=tuple(np.arange(self.num_dims) + 2))).detach().cpu().numpy().any(): import ipdb; ipdb.set_trace(context=5)
        assert (1 == final_mask.sum(dim=tuple(np.arange(self.num_dims) + 2))).detach().cpu().numpy().all()

        # pick weights and sum them
        tmp = torch.masked_select(self.weights, final_mask)
        if np.isnan(tmp.detach().cpu().numpy()).any(): import ipdb; ipdb.set_trace(context=5)
        tmp = tmp.view(bs, -1).sum(1, keepdim=True)
        return tmp