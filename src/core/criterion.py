import numpy as np
import torch.nn as nn
import torch


class WeightedBCE(object):

    def __init__(self, reduction, ones_weight, loss_weight):
        self.ones_weight = ones_weight
        self.criterion = nn.BCELoss(reduction=reduction)
        self.loss_weight = loss_weight

    def compute(self, pred_y, y, valid=None):
        loss = self.criterion(pred_y, y) # (Num_nodes, 4)
        valid = valid.view(pred_y.shape[0], pred_y.shape[1], pred_y.shape[2])

        if self.ones_weight > 1:
            ones_weight = np.ones_like(loss.detach().cpu().numpy())
            ones_weight[np.where(y.detach().cpu().numpy() == 1)] = self.ones_weight
            loss = torch.from_numpy(ones_weight).to(loss.device) * loss

        if valid is None:
            loss = self.loss_weight * (loss.mean())
        else:
            loss = self.loss_weight * (torch.sum(loss * valid) / torch.sum(valid))

        return loss


class WeightedBCEWithLogitsLoss(WeightedBCE):

    def __init__(self, reduction, **kwargs):
        super(WeightedBCEWithLogitsLoss, self).__init__(reduction, **kwargs)
        self.criterion = nn.BCEWithLogitsLoss(reduction=reduction)


class MSE(object):
    def __init__(self, loss_weight=1):

        self.criterion = nn.MSELoss()
        self.loss_weight = loss_weight

    def compute(self, pred_y, y):
        #TODO: Add valid support
        loss = self.criterion(pred_y, y) # (Num_nodes, 4)

        loss = self.loss_weight * loss

        return loss


class MAE(object):
    def __init__(self, loss_weight=1):

        self.criterion = nn.L1Loss()
        self.loss_weight = loss_weight

    def compute(self, pred_y, y):
        # TODO: Add valid support
        loss = self.criterion(pred_y, y) # (Num_nodes, 4)

        loss = self.loss_weight * loss

        return loss


class ExpectedLandmarkMSE(object):
    def __init__(self, loss_weight=1,
                 batch_size=2,
                 frame_size=128,
                 num_aux_graphs=6,
                 use_main_graph_only=False,
                 num_output_channels=4):

        self.loss_weight = loss_weight
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.num_aux_graphs = num_aux_graphs
        self.num_output_channels = num_output_channels

        if use_main_graph_only:
            self.grid_sizes = [frame_size]
            self.end_indices = [frame_size*frame_size]
        else:
            self.grid_sizes = [2**grid_idx for grid_idx in range(1, num_aux_graphs+1)] + [self.frame_size]
            # ending index of each flattened grid in the graph
            self.end_indices = np.cumsum([size**2 for size in self.grid_sizes]).tolist()

        # softmax layer applied on flattened image (HxW)
        self.softmax = torch.nn.Softmax(dim=1)
        self.use_main_graph_only = use_main_graph_only

    def compute(self, pred_y, y, valid):
        """
        pred_y is the logit output of model
        """
        
        # reshape to (N, nodes_in_batch, 4)
        pred_y = pred_y.view(self.batch_size, -1, self.num_output_channels)
        y = y.view(self.batch_size, -1, self.num_output_channels)
        valid = valid.view(self.batch_size, -1, self.num_output_channels)

        loss = 0
        start_idx = 0
        num_valid_samples = 0
        for idx, grid_size in enumerate(self.grid_sizes):
            end_idx = self.end_indices[idx]
            grid_shape = (self.batch_size,  # N
                          grid_size,  # H = Y
                          grid_size,  # W = X
                          self.num_output_channels)

            # extract groundtruth heatmap grid
            gt_heatmap = y[:, start_idx:end_idx, :].view(grid_shape)
            valid_subset = torch.mean(valid[:, start_idx:end_idx, :].permute(0, 2, 1), dim=-1).unsqueeze(-1)
            num_valid_samples = torch.sum(valid_subset, dim=0, keepdim=True)
            num_valid_samples[num_valid_samples == 0] = 1

            # calculate ground truth coordinates in order of (lvid_top, lvid_bot, lvpw, ivs)
            max_along_w, _ = torch.max(gt_heatmap, dim=-2)
            max_along_h, _ = torch.max(gt_heatmap, dim=-3)
            _, gt_h = torch.max(max_along_w, dim=-2)
            _, gt_w = torch.max(max_along_h, dim=-2)
            gt = torch.cat((gt_h.unsqueeze(2), gt_w.unsqueeze(2)), dim=2)

            # predicted probability heatmap using softmax layer
            softmaxed_heatmap = self.softmax(pred_y[:, start_idx:end_idx, :]).view(grid_shape)

            # initializing the mesh for x and y positions
            h = self.initialize_maps(grid_size, (1, -1, 1, 1)).to(pred_y.device)   # shape = 1,H,1,1
            w = self.initialize_maps(grid_size, (1, 1, -1, 1)).to(pred_y.device)   # shape = 1,1,W,1

            # getting the expected position of landmarks from the softmaxed predicted heatmap
            preds_h = torch.sum(torch.mul(softmaxed_heatmap, h), dim=(1, 2))  # shape = N,4
            preds_w = torch.sum(torch.mul(softmaxed_heatmap, w), dim=(1, 2))  # shape = N,4
            preds = torch.cat((preds_h.unsqueeze(2), preds_w.unsqueeze(2)), dim=2)  # shape = N,4,2

            # normalize coordinates to get meaningful MSE in grids of different coarsness
            preds, gt = (preds/grid_size, gt/grid_size)

            # computing the MSE loss from distance of predicted and gt landmarks
            loss_to_add = ((preds - gt)**2)
            loss_to_add *= valid_subset
            loss_to_add = torch.sum(loss_to_add, dim=0, keepdim=True) / num_valid_samples

            loss += loss_to_add.sum()

            # prep the start index for the next loop
            start_idx = end_idx

        return loss * self.loss_weight

    @staticmethod
    def initialize_maps(length, shape):
        """ creates an initial map for x and y elements with shape (1, length, 1, 1) or (1, 1, length, 1)
        length (int)
        shape (tuple):  (1,-1,1,1) for H,  (1,1,-1,1) for W,
        """
        line = torch.linspace(0, length - 1, length)
        map_init = torch.reshape(line, shape)
        return map_init


class HeatmapMSELoss(object):
    def __init__(self, reduction, ones_weight, loss_weight):
        self.ones_weight = ones_weight
        self.criterion = nn.MSELoss(reduction=reduction)
        self.loss_weight = loss_weight

    def compute(self, pred_y, y):
        # TODO: Add valid support
        loss = self.criterion(pred_y, y) # (Num_nodes, 4)

        if self.ones_weight > 1: #TODO FIX?
            ones_weight = np.ones_like(loss.detach().cpu().numpy())
            ones_weight[np.where(y.detach().cpu().numpy() > 0.05)] = self.ones_weight
            loss = torch.from_numpy(ones_weight).to(loss.device) * loss

        loss = self.loss_weight * (loss.mean())

        return loss
