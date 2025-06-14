import torch
import numpy as np, math
import torch.nn.functional as F
import torch.nn as nn


class Grad3d(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad3d, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


class Grad2d(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad2d, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred, y_true):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


class DisplacementRegularizer(torch.nn.Module):
    def __init__(self, energy_type):
        super().__init__()
        self.energy_type = energy_type

    def gradient_dx(self, fv):
        return (fv[:, 2:, 1:-1, 1:-1] - fv[:, :-2, 1:-1, 1:-1]) / 2

    def gradient_dy(self, fv):
        return (fv[:, 1:-1, 2:, 1:-1] - fv[:, 1:-1, :-2, 1:-1]) / 2

    def gradient_dz(self, fv):
        return (fv[:, 1:-1, 1:-1, 2:] - fv[:, 1:-1, 1:-1, :-2]) / 2

    def gradient_txyz(self, Txyz, fn):
        return torch.stack([fn(Txyz[:, i, ...]) for i in [0, 1, 2]], dim=1)

    def compute_gradient_norm(self, displacement, flag_l1=False):
        dTdx = self.gradient_txyz(displacement, self.gradient_dx)
        dTdy = self.gradient_txyz(displacement, self.gradient_dy)
        dTdz = self.gradient_txyz(displacement, self.gradient_dz)
        if flag_l1:
            norms = torch.abs(dTdx) + torch.abs(dTdy) + torch.abs(dTdz)
        else:
            norms = dTdx ** 2 + dTdy ** 2 + dTdz ** 2
        return torch.mean(norms) / 3.0

    def compute_bending_energy(self, displacement):
        dTdx = self.gradient_txyz(displacement, self.gradient_dx)
        dTdy = self.gradient_txyz(displacement, self.gradient_dy)
        dTdz = self.gradient_txyz(displacement, self.gradient_dz)
        dTdxx = self.gradient_txyz(dTdx, self.gradient_dx)
        dTdyy = self.gradient_txyz(dTdy, self.gradient_dy)
        dTdzz = self.gradient_txyz(dTdz, self.gradient_dz)
        dTdxy = self.gradient_txyz(dTdx, self.gradient_dy)
        dTdyz = self.gradient_txyz(dTdy, self.gradient_dz)
        dTdxz = self.gradient_txyz(dTdx, self.gradient_dz)
        return torch.mean(dTdxx ** 2 + dTdyy ** 2 + dTdzz ** 2 + 2 * dTdxy ** 2 + 2 * dTdxz ** 2 + 2 * dTdyz ** 2)

    def forward(self, disp, _):
        if self.energy_type == 'bending':
            energy = self.compute_bending_energy(disp)
        elif self.energy_type == 'gradient-l2':
            energy = self.compute_gradient_norm(disp)
        elif self.energy_type == 'gradient-l1':
            energy = self.compute_gradient_norm(disp, flag_l1=True)
        else:
            raise Exception('Not recognised local regulariser!')
        return energy


class NCC(torch.nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, device, win=None):
        super(NCC, self).__init__()
        self.win = win
        self.device = device

    def forward(self, y_pred, y_true):

        I = y_true
        J = y_pred

        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(self.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return 1 - torch.mean(cc)


class SpatialSegmentSmoothness(torch.nn.Module):
    def __init__(self, n_dims=3,
                 lambda_i=1.,
                 device="cuda:0"
                 ):
        super(SpatialSegmentSmoothness, self).__init__()
        self.n_dims = n_dims
        self.lambda_i = lambda_i
        self.device = device

    def forward(self, y_pred, y_true, contours):
        loss = 0
        segments_mask = 1. - contours

        for d in range(1, self.n_dims + 1):
            # Calculate the gradient (difference) along the current spatial dimension
            dCdx = y_pred.index_select(d + 1, torch.arange(1, y_pred.size(d + 1)).long().to(self.device)) \
                   - y_pred.index_select(d + 1, torch.arange(0, y_pred.size(d + 1) - 1).long().to(self.device))

            # Average across spatial dimensions and color channels
            loss += torch.mean(torch.abs(
                dCdx * segments_mask.index_select(d + 1, torch.arange(1, y_pred.size(d + 1)).long().to(self.device))))

        return loss


class Dice_mambamorph:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred, weight=None, return_per_loss=False, ignore_label=None):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        if weight is not None:
            B = len(y_true)
            assert len(weight) == B, "The length of data weights must be equal to the batch value."
            assert 0.99 < weight.sum().item() < 1.1, "The weights of data must sum to 1."
            weighted_loss = torch.tensor(0., device=y_true.device)
            per_loss = torch.zeros([B], dtype=torch.float32, device=y_true.device)
            for idx in range(B):
                top = 2 * (y_true[idx:idx + 1] * y_pred[idx:idx + 1]).sum(dim=vol_axes)
                bottom = torch.clamp((y_true[idx:idx + 1] + y_pred[idx:idx + 1]).sum(dim=vol_axes), min=1e-5)
                if ignore_label is not None:
                    item_dice = -torch.mean(top[:, ignore_label] / bottom[:, ignore_label])
                else:
                    item_dice = -torch.mean(top / bottom)
                weighted_loss += item_dice * weight[idx]
                per_loss[idx] = item_dice
            if return_per_loss:
                return weighted_loss, per_loss
            else:
                return weighted_loss
        else:
            top = 2 * (y_true * y_pred).sum(dim=vol_axes)
            bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
            if ignore_label is not None:
                dice = torch.mean(top[:, ignore_label] / bottom[:, ignore_label])
            else:
                dice = torch.mean(top / bottom)
        return 1 - dice

    def each_dice(self, y_true, y_pred, ignore_label=None):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        if ignore_label is not None:
            dice = top[:, ignore_label] / bottom[:, ignore_label]
        else:
            dice = top / bottom
        return 1 - dice
