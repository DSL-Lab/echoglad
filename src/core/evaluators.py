import torch
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error
import torchvision
import matplotlib.pyplot as plt


class BinaryAccuracyEvaluator(object):

    def __init__(self,  logger):

        self.score = 0.
        self.count = 0

    def reset(self):

        self.score = 0.
        self.count = 0

    def update(self, y_pred, y_true):

        self.count += 1
        self.score += accuracy_score(y_true=y_true, y_pred=y_pred > 0.5)

    def compute(self):

        return self.score / self.count


class MSEEvaluator(object):

    def __init__(self,  logger):

        self.score_per_class = None  # shape N,4

    def reset(self):

        self.score_per_class = None  # shape N,4

    def update(self, y_pred, y_true):
        """
        y_true: true binary labels for each node (num_nodes, num_classes)
        y_pred: sigmoid outputs (num_nodes, num_classes)
        """
        y_pred = y_pred.numpy()
        y_true = y_true.numpy()

        # per class accuracy calculation
        if self.score_per_class is None:
            self.score_per_class = self.compute_per_class(y_pred, y_true)
        else:
            # becomes a numpy array with shape of (N,4)
            self.score_per_class = np.append(self.score_per_class, self.compute_per_class(y_pred, y_true), axis=0)

    def compute(self):
        """
        computes average of MSE across landmarks
        """

        return self.get_per_class_score().mean() # first average across the batch, then across  4 landmark errors

    def get_per_class_score(self):
        """
        returns the average MSEs for all classes
        """

        return self.score_per_class.mean(axis=0)

    def get_last(self):
        return self.score_per_class[-1, :].mean()

    @staticmethod
    def compute_per_class(y_pred, y_true):
        """ computes MSE score for each of the landmarks separately
        score_per_class (array): shape (num_classes,)
        """

        score_per_class = []
        for idx in range(y_true.shape[-1]):
            score_per_class.append(mean_squared_error(y_true=y_true[:, idx], y_pred=y_pred[:, idx]))

        return np.asarray(score_per_class).reshape((1,-1))


class BalancedBinaryAccuracyEvaluator(object):

    def __init__(self,  logger):

        self.score_per_class = None # shape N,4

    def reset(self):

        self.score_per_class = None

    def update(self, y_pred, y_true, valid):
        """
        y_true: true binary labels for each node (num_nodes, num_classes)
        y_pred: sigmoid outputs (num_nodes, num_classes)
        """
        y_pred = y_pred.numpy()
        y_true = y_true.numpy()

        # per class accuracy calculation
        if self.score_per_class is None:
            self.score_per_class = self.compute_per_class(y_pred, y_true, valid)
        else:
            # becomes a numpy array with shape of (N,4)
            self.score_per_class = np.append(self.score_per_class, self.compute_per_class(y_pred,
                                                                                          y_true,
                                                                                          valid), axis=0)

    def compute(self):
        """
        computes average of balanced binary accuracies across landmarks
        """

        return self.score_per_class.mean(axis=0).mean() # first average across the batch, then across  4 landmark errors

    def get_per_class_score(self):
        """
        returns the average accuracy for all classes
        """

        return self.score_per_class.mean(axis=0)

    def get_last(self):
        return self.score_per_class[-1,:].mean()

    @staticmethod
    def compute_per_class(y_pred, y_true, valid):
        """ computes balanced binary accuracy for each of the landmarks separately
        score_per_class (array): shape (num_classes,)
        """

        score_per_class = []
        for idx in range(y_true.shape[-1]):
            if torch.count_nonzero(valid[:, idx]) > 0:
                score_per_class.append(balanced_accuracy_score(y_true=y_true[:, idx][valid[:, idx] > 0],
                                                               y_pred=y_pred[:, idx][valid[:, idx] > 0] > 0.5))
            else:
                score_per_class.append(0)

        return np.asarray(score_per_class).reshape((1, -1))


class LandmarkErrorEvaluator(object):
    def __init__(self, logger, batch_size, frame_size, use_coord_graph):

        self.errors = {'lvid': [],
                       'ivs': [],
                       'lvpw': []}
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.use_coord_graph = use_coord_graph

    def reset(self):

        self.errors = {'lvid': [],
                       'ivs': [],
                       'lvpw': []}

    def update(self, y_pred, y_true):

        nodes_in_batch = y_pred.size(0) // self.batch_size

        y_pred = y_pred.view(self.batch_size, nodes_in_batch, 4).detach().cpu().numpy()
        y_true = y_true.view(self.batch_size, nodes_in_batch, 4).detach().cpu().numpy()

        lvid_err = []
        lvpw_err = []
        ivs_err = []

        for i in range(self.batch_size):
            pred_heatmap = torch.tensor(y_pred[i, -self.frame_size*self.frame_size:, :]).view(self.frame_size,
                                                                                              self.frame_size,
                                                                                              4)
            gt_heatmap = torch.tensor(y_true[i, -self.frame_size*self.frame_size:, :]).view(self.frame_size,
                                                                                            self.frame_size,
                                                                                            4)

            # lvid_top, lvid_bot, lvpw, ivs
            gt_x = torch.argmax(torch.argmax(gt_heatmap, 0), 0)
            gt_y = torch.argmax(torch.argmax(gt_heatmap, 1), 0)
            gt_lvid = self.get_pixel_length(gt_x, gt_y, 0, 1)
            gt_ivs = self.get_pixel_length(gt_x, gt_y, 0, 3)
            gt_lvpw = self.get_pixel_length(gt_x, gt_y, 2, 1)

            preds_x = torch.argmax(torch.argmax(pred_heatmap, 0), 0)
            preds_y = torch.argmax(torch.argmax(pred_heatmap, 1), 0)
            pred_lvid = self.get_pixel_length(preds_x, preds_y, 0, 1)
            pred_ivs = self.get_pixel_length(preds_x, preds_y, 0, 3)
            pred_lvpw = self.get_pixel_length(preds_x, preds_y, 2, 1)

            lvid_err.append(torch.abs(pred_lvid - gt_lvid))
            ivs_err.append(torch.abs(pred_ivs - gt_ivs))
            lvpw_err.append(torch.abs(pred_lvpw - gt_lvpw))

        self.errors['lvid'].append(np.mean(lvid_err))
        self.errors['ivs'].append(np.mean(ivs_err))
        self.errors['lvpw'].append(np.mean(lvpw_err))

    def compute(self):
        """
        compute the mean of all the recorded
        """

        temp = dict()
        temp['ivs_w'] = np.asarray(self.errors['ivs']).mean()
        temp['lvid_w'] = np.asarray(self.errors['lvid']).mean()
        temp['lvpw_w'] = np.asarray(self.errors['lvpw']).mean()

        return temp

    def get_last(self):
        temp = dict()
        temp['ivs_w'] = self.errors['ivs'][-1]
        temp['lvid_w'] = self.errors['lvid'][-1]
        temp['lvpw_w'] = self.errors['lvpw'][-1]

        return temp

    def get_heatmaps(self, y_pred):
        titles = ['lvid_top', 'lvid_bot', 'lvpw', 'ivs']
        nodes_in_batch = y_pred.size(0) // self.batch_size
        y_pred = y_pred.view(self.batch_size, nodes_in_batch, 4).detach().cpu().numpy()
        hms = y_pred[0, -self.frame_size*self.frame_size:, :].reshape(self.frame_size, self.frame_size, 4)
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        for i in range(4):
            axs[i].imshow(hms[:, :, i], cmap='cool')
            axs[i].set_title(titles[i])
            axs[i].axis('off')
        return fig

    @staticmethod
    def get_pixel_length(x, y, i, j):
        return torch.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)


class LandmarkExpectedCoordiantesEvaluator(object):
    """
    to locate the landmarks by finding the expected value of a softmaxed heatmap and calcualte how far they are from gt
    """
    def __init__(self, logger, batch_size, frame_size, use_coord_graph):

        self.coordinate_errors = {'ivs': [],
                                  'lvid_top': [],
                                  'lvid_bot': [],
                                  'lvpw': []}
        self.valid_errors = {'ivs': [],
                             'lvid_top': [],
                             'lvid_bot': [],
                             'lvpw': []}
        self.width_MAE = {'lvid': [],
                          'ivs': [],
                          'lvpw': []}
        self.width_MPE = {'lvid': [],
                          'ivs': [],
                          'lvpw': []}
        self.detailed_performance = {}
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.use_coord_graph = use_coord_graph

    def reset(self):

        self.coordinate_errors = {'ivs': [],
                                  'lvid_top': [],
                                  'lvid_bot': [],
                                  'lvpw': []}
        self.valid_errors = {'ivs': [],
                             'lvid_top': [],
                             'lvid_bot': [],
                             'lvpw': []}
        self.width_MAE = {'lvid': [],
                          'ivs': [],
                          'lvpw': []}
        self.width_MPE = {'lvid': [],
                          'ivs': [],
                          'lvpw': []}
        self.detailed_performance.clear()

    @staticmethod
    def initialize_maps(length, shape):
        '''
            creates an initial map for x and y elements with shape (1, length, 1, 1) or (1, 1, length, 1)
        '''
        line = torch.linspace(0, length - 1, length)
        map_init = torch.reshape(line, shape)
        return map_init

    def update(self, y_pred, y_true, pix2mm_x, pix2mm_y, valid):
        """
        shapes are (batch_size x nodes_in_batch, num_landmarks) = (N x nodes_in_batch, 4)
        y_pred: tensor of logits for node prediction before softmax.
        y_true: tensor of node ground truth
        """
        self.detailed_performance.clear()

        if self.use_coord_graph:

            preds = y_pred.view(y_pred.shape[0] // 4, 4, 2)
            gt = y_true.view(y_true.shape[0] // 4, 4, 2)
            gt_h = gt[:, :, 0]
            gt_w = gt[:, :, 1]
            preds_h = preds[:, :, 0]
            preds_w = preds[:, :, 1]

        else:
            # reshape to (N, nodes_in_batch, 4)
            y_pred = y_pred.view(self.batch_size, -1, y_pred.shape[-1]).detach()
            y_true = y_true.view(self.batch_size, -1, y_true.shape[-1]).detach()
            valid = valid.view(self.batch_size, -1, valid.shape[-1])
            valid_subset = torch.mean(valid[:,
                                      -self.frame_size*self.frame_size:,
                                      :].permute(0, 2, 1), dim=-1)
            num_valid_samples = torch.sum(valid_subset, dim=0, keepdim=True)

            self.valid_errors['lvid_top'].append((num_valid_samples[0, 0] > 0).item())
            self.valid_errors['lvid_bot'].append((num_valid_samples[0, 1] > 0).item())
            self.valid_errors['lvpw'].append((num_valid_samples[0, 2] > 0).item())
            self.valid_errors['ivs'].append((num_valid_samples[0, 3] > 0).item())

            num_valid_samples[num_valid_samples == 0] = 1

            # shape (N, 224, 224, 4)
            gt_heatmap = y_true[:, -self.frame_size * self.frame_size:, :].view(self.batch_size, # N
                                                                                self.frame_size, # H = Y
                                                                                self.frame_size, # W = X
                                                                                y_true.shape[-1])

            # lvid_top, lvid_bot, lvpw, ivs
            max_along_w, _ = torch.max(gt_heatmap, dim=-2)
            max_along_h, _ = torch.max(gt_heatmap, dim=-3)
            _, gt_h = torch.max(max_along_w, dim=-2)
            _, gt_w = torch.max(max_along_h, dim=-2)
            gt = torch.cat((gt_h.unsqueeze(2), gt_w.unsqueeze(2)), dim=2)  # shape=(N,4,2)

            # probability heatmap using softmax layer
            pred_heatmap = self.get_softmaxed_heatmap(y_pred)

            # initializing the mesh for x and y positions
            h = self.initialize_maps(self.frame_size, (1,-1,1,1))#.to(pred_heatmap.device)  # shape = 1,H,1,1
            w = self.initialize_maps(self.frame_size, (1,1,-1,1))#.to(pred_heatmap.device)  # shape = 1,1,W,1

            # getting the expected position of landmarks
            preds_h = torch.sum(torch.mul(pred_heatmap, h), dim=(1, 2))  # shape = N,4
            preds_w = torch.sum(torch.mul(pred_heatmap, w), dim=(1, 2))  # shape = N,4
            preds = torch.cat((preds_h.unsqueeze(2), preds_w.unsqueeze(2)), dim=2)  # shape = (N,4,2)

        # calculating error between pred/gt coordinates
        landmark_errors = self.get_pixel_length(gt_w, gt_h,
                                                preds_w, preds_h,
                                                pix2mm_x.unsqueeze(1),
                                                pix2mm_y.unsqueeze(1)).numpy()  # shape (N, 4)

        landmark_errors *= valid_subset.numpy()
        landmark_errors = np.squeeze(np.sum(landmark_errors, axis=0) / num_valid_samples.numpy())

        self.coordinate_errors['lvid_top'].append(landmark_errors[0])
        self.coordinate_errors['lvid_bot'].append(landmark_errors[1])
        self.coordinate_errors['lvpw'].append(landmark_errors[2])
        self.coordinate_errors['ivs'].append(landmark_errors[3])

        # calculating error between pred/gt widths
        widths = self.calculate_widths(preds, gt, pix2mm_x, pix2mm_y)  # shape (N)
        ivs_err, lvid_err, lvpw_err = self.calculate_width_MAE(widths)  # shape (N)

        lvid_err *= valid_subset[:, 0] * valid_subset[:, 1] / torch.min(num_valid_samples[0, 0], num_valid_samples[0, 1])
        ivs_err *= valid_subset[:, 3] / num_valid_samples[0, 3]
        lvpw_err *= valid_subset[:, 2] / num_valid_samples[0, 2]

        self.width_MAE['ivs'].append(ivs_err.sum().item())
        self.width_MAE['lvid'].append(lvid_err.sum().item())
        self.width_MAE['lvpw'].append(lvpw_err.sum().item())

        ivs_err, lvid_err, lvpw_err = self.calculate_width_MPE(widths)  # shape (N)

        lvid_err *= valid_subset[:, 0] * valid_subset[:, 1] / torch.min(num_valid_samples[0, 0], num_valid_samples[0, 1])
        ivs_err *= valid_subset[:, 3] / num_valid_samples[0, 3]
        lvpw_err *= valid_subset[:, 2] / num_valid_samples[0, 2]

        self.width_MPE['ivs'].append(ivs_err.sum().item())
        self.width_MPE['lvid'].append(lvid_err.sum().item())
        self.width_MPE['lvpw'].append(lvpw_err.sum().item())

        # recording the performance to be accessed later
        coordinates = {
            'pred_ivs': preds[:,3], 'pred_lvid_top': preds[:,0], 'pred_lvid_bot': preds[:,1], 'pred_lvpw': preds[:,2],
            'gt_ivs': gt[:, 3], 'gt_lvid_top': gt[:, 0], 'gt_lvid_bot': gt[:, 1], 'gt_lvpw': gt[:, 2]
        }
        self.detailed_performance = {'widths': widths, 'coordinates': coordinates}

    def calculate_widths(self, preds, gt, pix2mm_x, pix2mm_y):
        """
        input shapes are N,4,2
        output values of the dictionary are tensors with shape of (N)
        """

        widths = {"pred_ivs_mm": self.get_pixel_length(preds[:,3,1], preds[:,3,0], preds[:,0,1], preds[:,0,0], pix2mm_x, pix2mm_y),
                  "pred_lvid_mm": self.get_pixel_length(preds[:,0,1], preds[:,0,0], preds[:,1,1], preds[:,1,0], pix2mm_x, pix2mm_y),
                  "pred_lvpw_mm": self.get_pixel_length(preds[:, 1, 1], preds[:, 1, 0], preds[:, 2, 1], preds[:, 2, 0], pix2mm_x, pix2mm_y),
                  "gt_ivs_mm": self.get_pixel_length(gt[:,3,1], gt[:,3,0], gt[:,0,1], gt[:,0,0], pix2mm_x, pix2mm_y),
                  "gt_lvid_mm": self.get_pixel_length(gt[:, 0, 1], gt[:, 0, 0], gt[:, 1, 1], gt[:, 1, 0], pix2mm_x,pix2mm_y),
                  "gt_lvpw_mm": self.get_pixel_length(gt[:, 1, 1], gt[:, 1, 0], gt[:, 2, 1], gt[:, 2, 0], pix2mm_x,pix2mm_y)}

        return widths

    def calculate_width_MAE(self, widths):
        """
        calculate the absolute errors between predicted and gt width of the landmarks. shape (N)
        """
        ivs_err = torch.abs(widths['pred_ivs_mm'] - widths['gt_ivs_mm'])
        lvid_err = torch.abs(widths['pred_lvid_mm'] - widths['gt_lvid_mm'])
        lvpw_err = torch.abs(widths['pred_lvpw_mm'] - widths['gt_lvpw_mm'])
        return ivs_err, lvid_err, lvpw_err

    def calculate_width_MPE(self, widths):
        """
        calculate the percentage errors between predicted and gt width of the landmarks. shape (N)
        """
        ivs_err = 100 * torch.abs(widths['pred_ivs_mm'] - widths['gt_ivs_mm']) / widths['gt_ivs_mm']
        lvid_err = 100 * torch.abs(widths['pred_lvid_mm'] - widths['gt_lvid_mm']) / widths['gt_lvid_mm']
        lvpw_err = 100 * torch.abs(widths['pred_lvpw_mm'] - widths['gt_lvpw_mm']) / widths['gt_lvpw_mm']
        return ivs_err, lvid_err, lvpw_err

    def compute(self):
        """
        compute the mean of all the recorded
        """

        temp = dict()
        temp['lvid_top'] = np.asarray(self.coordinate_errors['lvid_top']).sum() / np.count_nonzero( np.asarray(self.valid_errors['lvid_top']))
        temp['lvid_bot'] = np.asarray(self.coordinate_errors['lvid_bot']).sum() / np.count_nonzero( np.asarray(self.valid_errors['lvid_bot']))
        temp['lvpw'] = np.asarray(self.coordinate_errors['lvpw']).sum() / np.count_nonzero( np.asarray(self.valid_errors['lvpw']))
        temp['ivs'] = np.asarray(self.coordinate_errors['ivs']).sum() / np.count_nonzero( np.asarray(self.valid_errors['ivs']))

        temp['ivs_w'] = np.asarray(self.width_MAE['ivs']).sum() / np.count_nonzero( np.asarray(self.valid_errors['ivs']))
        temp['lvid_w'] = np.asarray(self.width_MAE['lvid']).sum() / np.count_nonzero(np.logical_and(np.asarray(self.valid_errors['lvid_top']), np.asarray(self.valid_errors['lvid_bot'])))
        temp['lvpw_w'] = np.asarray(self.width_MAE['lvpw']).sum() / np.count_nonzero( np.asarray(self.valid_errors['lvpw']))

        temp['ivs_mpe'] = np.asarray(self.width_MPE['ivs']).sum() / np.count_nonzero( np.asarray(self.valid_errors['ivs']))
        temp['lvid_mpe'] = np.asarray(self.width_MPE['lvid']).sum() / np.count_nonzero(np.logical_and(np.asarray(self.valid_errors['lvid_top']), np.asarray(self.valid_errors['lvid_bot'])))
        temp['lvpw_mpe'] = np.asarray(self.width_MPE['lvpw']).sum() / np.count_nonzero( np.asarray(self.valid_errors['lvpw']))

        return temp

    def get_sum_of_width_MAE(self):
        """
        returns the sum of widths mean absolute errors
        """
        temp = self.compute()
        return sum([value for k, value in temp.items() if k in ['ivs_w', 'lvid_w', 'lvpw_w']])

    def get_sum_of_width_MPE(self):
        """
        returns the sum of widths' mean percent errors
        """
        temp = self.compute()
        return sum([value for k, value in temp.items() if k in ['ivs_mpe', 'lvid_mpe', 'lvpw_mpe']])

    def get_last(self):
        temp = dict()
        temp['lvid_top'] = self.coordinate_errors['lvid_top'][-1]
        temp['lvid_bot'] = self.coordinate_errors['lvid_bot'][-1]
        temp['lvpw'] = self.coordinate_errors['lvpw'][-1]
        temp['ivs'] = self.coordinate_errors['ivs'][-1]

        temp['ivs_w'] = self.width_MAE['ivs'][-1]
        temp['lvid_w'] = self.width_MAE['lvid'][-1]
        temp['lvpw_w'] = self.width_MAE['lvpw'][-1]

        temp['ivs_mpe'] = self.width_MPE['ivs'][-1]
        temp['lvid_mpe'] = self.width_MPE['lvid'][-1]
        temp['lvpw_mpe'] = self.width_MPE['lvpw'][-1]

        return temp

    def get_predictions(self):
        """
        returns a dictionary of lists containing detailed performance of model in the current iteration.
        each value is in list format with length of batch_size
        """
        return self.detailed_performance

    def get_softmaxed_heatmap(self, y_pred):

        """
        y_pred shape is (N, HxW, 4)
        """
        softmax = torch.nn.Softmax(dim=1)
        pred_heatmap = softmax(y_pred[:, -self.frame_size * self.frame_size:, :]).view(-1,  # N
                                                                                       self.frame_size,  # H = Y
                                                                                       self.frame_size,  # W = X
                                                                                       y_pred.shape[-1])
        return pred_heatmap

    def get_heatmaps(self, x, landmark_preds, landmark_y, coord_preds, pix2mm_x, pix2mm_y):

        # get the input image for the first sample in the batch
        x = x[0,0]  # (H, W)

        # get the gt heatmap for the first sample in batch
        landmark_y = landmark_y.view(self.batch_size, -1, 4)
        landmark_y = landmark_y[0, -self.frame_size * self.frame_size:, :].view(self.frame_size,  # H = Y
                                                                                self.frame_size,  # W = X
                                                                                4).numpy()

        # get the expected coordinates from softmaxed heatmap of the first sample in batch
        landmark_preds = landmark_preds.view(self.batch_size, -1, 4)  # (N, num_nodes, 4)
        hms = self.get_softmaxed_heatmap(landmark_preds[0:1])[0] # (H, W, 4)
        hms = hms / hms.max(dim=0, keepdim=True)[0].max(dim=1, keepdim=True)[0]  # normalize heatmaps per channel

        # get PIL images with input frame with overlayed prediction / gt
        img = {'pred_img': self.create_overlay_image(x, hms),
               'gt_img': self.create_overlay_image(x, landmark_y)}

        widths = self.detailed_performance['widths']
        widths = {key: value[0] for key,value in widths.items()}
        ivs_mae, lvid_mae, lvpw_mae = self.calculate_width_MAE(widths)
        ivs_mpe, lvid_mpe, lvpw_mpe = self.calculate_width_MPE(widths)  # shape (N)
        coordinates = self.detailed_performance['coordinates']
        coordinates = {key: value[0].tolist() for key,value in coordinates.items()} #TODO integer

        if coord_preds is not None:
            # get the coordinate predictions for the first sample in batch
            coord_preds = coord_preds.view(self.batch_size, -1, 2)[0]
            coordinates.update({
                'coord_ivs': list(coord_preds[3].numpy()),
                'coord_lvid_top': list(coord_preds[0].numpy()),
                'coord_lvid_bot': list(coord_preds[1].numpy()),
                'coord_lvpw': list(coord_preds[2].numpy()),
            })
            widths.update({
                "coord_ivs_mm": self.get_pixel_length(coord_preds[3, 0], coord_preds[3, 1],
                                                      coord_preds[0, 0], coord_preds[0, 1],
                                                      pix2mm_x[0], pix2mm_y[0]),
                "coord_lvid_mm": self.get_pixel_length(coord_preds[0, 0], coord_preds[0, 1], coord_preds[1, 0], coord_preds[1, 1],
                                                      pix2mm_x[0], pix2mm_y[0]),
                "coord_lvpw_mm": self.get_pixel_length(coord_preds[1, 0], coord_preds[1, 1], coord_preds[2, 0], coord_preds[2, 1],
                                                      pix2mm_x[0], pix2mm_y[0]),

            })

                # get img overlay for the coordinates
            img.update({'coord_img': self.create_overlay_image(x, torch.zeros_like(hms))})

            fig, axs = plt.subplots(1, 3, figsize=(12, 5)) # todo check
            group = ['pred', 'coord', 'gt']
        else:
            fig, axs = plt.subplots(1, 2, figsize=(8, 5))
            group = ['pred', 'gt']

        linewidth = 1.5
        markersize = 4
        landmarks = ['lvid_top', 'lvid_bot', 'lvpw', 'ivs']
        for i, gp in enumerate(group):
            # plot the image
            axs[i].imshow(img[f'{gp}_img'])

            # overlay landmarks points
            for landmark in landmarks:
                name = f'{gp}_{landmark}'
                axs[i].plot(coordinates[name][1],
                            coordinates[name][0], 
                            marker='x', color='white', markersize=markersize)

            # overlay widths lines
            axs[i].plot([coordinates[f'{gp}_ivs'][1], coordinates[f'{gp}_lvid_top'][1]],
                        [coordinates[f'{gp}_ivs'][0], coordinates[f'{gp}_lvid_top'][0]],
                        [coordinates[f'{gp}_lvid_bot'][1], coordinates[f'{gp}_lvpw'][1]],
                        [coordinates[f'{gp}_lvid_bot'][0], coordinates[f'{gp}_lvpw'][0]],
                        color="dodgerblue", linewidth=linewidth)
            axs[i].plot([coordinates[f'{gp}_lvid_top'][1], coordinates[f'{gp}_lvid_bot'][1]],
                        [coordinates[f'{gp}_lvid_top'][0], coordinates[f'{gp}_lvid_bot'][0]], 
                        color="red", linewidth=linewidth)

            # add titles
            axs[i].set_title(f"{gp}: [{widths[f'{gp}_ivs_mm']:.1f},"
                             f" {widths[f'{gp}_lvid_mm']:.1f},"
                             f" {widths[f'{gp}_lvpw_mm']:.1f}]")    #TODO add the calculated widths

        # some visual configs for the figure
        plt.setp(axs[1].get_yticklabels(), visible=False)  # hide y ticks for the right plot
        fig.tight_layout()

        # add super title for the figure
        fig.suptitle(
            f"MAE [mm] | IVS: {ivs_mae:.1f} | LVID: {lvid_mae:.1f} | LVPW: {lvpw_mae:.1f}"
            f"\n"
            f"MPE [%] | IVS: {ivs_mpe:.1f} | LVID: {lvid_mpe:.1f} | LVPW: {lvpw_mpe:.1f}"
        )

        return fig

    def create_overlay_image(self, x, hms):
        """
        x is grayscale image. tensor with shape (H,W)
        hms is the matrix of heatmaps. tensor with shape (H,W,Channels)

        returns a PIL Image object
        """
        cmap = plt.get_cmap('hsv')
        img = torch.zeros(3, self.frame_size, self.frame_size)
        # for grayscale
        color = torch.FloatTensor((0.8,.8,.8)).reshape([-1, 1, 1])
        img = torch.max(img, color * x)  # max in case of overlapping position of joints
        # for heatmaps
        colors_rgb = [(0,1,1), (1,0.7,0.9), (0,1,0), (1,0,0)]
        # C = 4
        for i, color_rgb in enumerate(colors_rgb):
            # color = torch.FloatTensor(cmap(i * cmap.N // C)[:3]).reshape([-1, 1, 1])
            color = torch.FloatTensor(color_rgb).reshape([-1, 1, 1])
            img = torch.max(img, color * hms[:,:,i])  # max in case of overlapping position of joints
        img = torchvision.transforms.ToPILImage()(img)        # fig, axs = plt.subplots(1, 4, figsize=(16, 4))

        return img

    @staticmethod
    def get_pixel_length(x0, y0, x1, y1, pix2mm_x, pix2mm_y):
        return torch.sqrt(((x0 - x1)*pix2mm_x) ** 2 + ((y0 - y1)*pix2mm_y) ** 2)