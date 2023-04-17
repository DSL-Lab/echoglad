import torch
import time
from datetime import timedelta
from src.utils import util
from src.builders import model_builder, dataloader_builder, checkpointer_builder, \
    optimizer_builder, criterion_builder, scheduler_builder, \
    meter_builder, evaluator_builder, dataset_builder
import wandb
from tqdm import tqdm
import torch_geometric
import matplotlib.pyplot as plt
import pandas as pd
import os.path as osp


class BaseEngine(object):

    def __init__(self, config, logger, save_dir):
        # Assign a logger and save dir
        self.logger = logger
        self.save_dir = save_dir

        # Load configurations
        self.model_config = config['model']
        self.train_config = config['train']
        self.eval_config = config['eval']
        self.data_config = config['data']

        # Determine which device to use
        seed = self.train_config['seed']
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)  # set the manual seed for torch
            self.device = torch.device("cuda")
        else:
            torch.manual_seed(seed)  # set the manual seed for torch
            self.device = torch.device("cpu")
        self.logger.info(f"Seed is {seed}")

        if self.device == 'cpu':
            self.logger.warn('GPU is not available.')
        else:
            self.logger.warn('{} GPU(s) is/are available.'.format(
                torch.cuda.device_count()))

        # Set up Wandb if required
        if config['train']['use_wandb']:
            wandb.init(project=config['train']['wand_project_name'],
                       name=None if config['train']['wandb_run_name'] == '' else config['train']['wandb_run_name'],
                       config=config,
                       mode=config['train']['wandb_mode'])

            # define our custom x axis metric
            wandb.define_metric("batch_train/step")
            wandb.define_metric("batch_valid/step")
            wandb.define_metric("epoch")
            # set all other metrics to use the corresponding step
            wandb.define_metric("batch_train/*", step_metric="batch_train/step")
            wandb.define_metric("batch_valid/*", step_metric="batch_valid/step")
            wandb.define_metric("epoch/*", step_metric="epoch")
            wandb.define_metric("lr", step_metric="epoch")
        self.wandb_log_steps = config['train'].get('wandb_log_steps', 1000)

    def run(self):
        pass

    def evaluate(self):
        pass


class Engine(BaseEngine):

    def __init__(self, config, logger, save_dir):
        super(Engine, self).__init__(config, logger, save_dir)

    def _build(self, mode='train'):

        # Create datasets
        dataset = dataset_builder.build(data_config=self.data_config, logger=self.logger)

        # Build a dataloader
        self.dataloaders = dataloader_builder.build(dataset=dataset,
                                                    train_config=self.train_config,
                                                    logger=self.logger,
                                                    use_data_parallel=True if torch.cuda.device_count() > 1 else False)

        # Flag indicating if coordinate graph is used for the main task
        self.use_coordinate_graph = self.data_config.get('use_coordinate_graph', False)
        self.use_connection_nodes = self.data_config.get('use_connection_nodes', False)
        self.use_main_graph_only = self.data_config.get('use_main_graph_only', False)

        # Build a model
        # add other required configs to model config
        self.frame_size = self.data_config['transform']['image_size']
        self.num_output_channels = 4
        self.model_config['landmark'].update({'frame_size': self.data_config['transform']['image_size']})
        self.model_config['landmark'].update({'num_aux_graphs': self.data_config['num_aux_graphs']})
        self.model_config['landmark'].update({'use_coordinate_graph': self.use_coordinate_graph})
        self.model_config['landmark'].update({'use_connection_nodes': self.use_connection_nodes})
        self.model_config['landmark'].update({'use_main_graph_only': self.use_main_graph_only})
        self.model_config['landmark'].update({'num_output_channels': self.num_output_channels})
        self.model = model_builder.build(
            self.model_config, self.logger)

        # Use multi GPUs if available
        if torch.cuda.device_count() > 1:
            for model_key in self.model:
                if model_key == 'landmark':
                    self.model[model_key] = torch_geometric.nn.DataParallel(self.model[model_key])
                else:
                    self.model[model_key] = torch.nn.DataParallel(self.model[model_key])

        for model_name in self.model.keys():
            self.model[model_name].to(self.device)

        # Build the optimizer
        self.optimizer = optimizer_builder.build(
            self.train_config['optimizer'], self.model, self.logger)

        # Build the scheduler
        self.scheduler = scheduler_builder.build(
            self.train_config, self.optimizer, self.logger)

        # Build the criterion
        # add other required configs
        self.train_config['criterion'].update({'frame_size': self.data_config['transform']['image_size'],
                                               'num_aux_graphs': self.data_config['num_aux_graphs'],
                                               'batch_size': self.train_config['batch_size'],
                                               'use_coordinate_graph': self.use_coordinate_graph,
                                               'use_main_graph_only': self.use_main_graph_only,
                                               'num_output_channels': self.num_output_channels})
        self.criterion = criterion_builder.build(config=self.train_config['criterion'], logger=self.logger)

        # Build the loss meter
        self.loss_meter = meter_builder.build(self.logger)

        # Build the evaluators
        # add other required configs
        self.eval_config.update({'frame_size': self.data_config['transform']['image_size'],
                                 'batch_size': self.train_config['batch_size'],
                                 'use_coordinate_graph': self.use_coordinate_graph})
        self.evaluators = evaluator_builder.build(self.eval_config, logger=self.logger)

        # Build a checkpointer
        self.checkpointer = checkpointer_builder.build(
            self.save_dir, self.logger, self.model, self.optimizer,
            self.scheduler, self.eval_config['standard'], best_mode='min')  #TODO make best_mode configurable
        checkpoint_path = self.model_config.get('checkpoint_path', '')
        self.misc = self.checkpointer.load(
            mode, checkpoint_path, use_latest=False)

    def run(self):
        start_epoch, num_steps = 0, 0
        num_epochs = self.train_config.get('num_epochs', 100)
        checkpoint_step = self.train_config.get('checkpoint_step', 1000)

        self._build(mode='train')

        self.logger.info(
            'Train for {} epochs starting from epoch {}'.format(num_epochs, start_epoch))

        # Start training
        for epoch in range(start_epoch, start_epoch + num_epochs):
            util.reset_evaluators(self.evaluators)
            self.loss_meter.reset()

            train_start = time.time()
            num_steps = self._train_one_epoch(epoch, num_steps, checkpoint_step)
            train_time = time.time() - train_start

            # print a summary of the training epoch
            self.log_summary("Training", epoch, train_time)
            self.log_wandb({'loss_total': self.loss_meter.avg}, {"epoch": epoch}, mode='epoch/train')

            if self.train_config['lr_schedule']['name'] == 'multi':
                self.scheduler.step()
            self.loss_meter.reset()
            util.reset_evaluators(self.evaluators)

            # Evaluate
            # if epoch - start_epoch > 0.0 * num_epochs:
            train_start = time.time()
            self._evaluate_once(epoch, num_steps)
            validation_time = time.time() - train_start

            # step lr scheduler with the sum of landmark width errors
            if self.train_config['lr_schedule']['name'] == 'reduce_lr_on_plateau':
                self.scheduler.step(self.evaluators["landmarkcoorderror"].get_sum_of_width_MAE())

            self.checkpointer.save(epoch,
                                   num_steps,
                                   self.evaluators["landmarkcoorderror"].get_sum_of_width_MPE(),
                                   best_mode='min')
            self.log_wandb({'loss_total': self.loss_meter.avg}, {"epoch": epoch}, mode='epoch/valid')
            # print a summary of the validation epoch
            self.log_summary("Validation", epoch, validation_time)

    def _train_one_epoch(self, epoch, num_steps, checkpoint_step):
        dataloader = self.dataloaders['train']

        for model_name in self.model.keys():
            self.model[model_name].train()

        epoch_steps = len(dataloader)
        data_iter = iter(dataloader)
        iterator = tqdm(range(epoch_steps), dynamic_ncols=True)
        for i in iterator:
            data_batch = next(data_iter)

            # Process each sample in the batch
            if type(data_batch) == list:

                for indx in range(len(data_batch)):
                    data_batch[indx].to(self.device)

                input_frames = torch.cat([graph.x for graph in data_batch])
                node_landmark_y = torch.cat([graph.y for graph in data_batch])
                node_landmark_valid = torch.cat([graph.valid_labels for graph in data_batch])
                pix2mm_x = torch.hstack([graph.pix2mm_x for graph in data_batch])
                pix2mm_y = torch.hstack([graph.pix2mm_y for graph in data_batch])

                node_coord_y = None
                if self.use_coordinate_graph:
                    node_coord_y = torch.cat([graph.node_coord_y for graph in data_batch])

            else:
                data_batch.to(self.device)
                input_frames = data_batch.x
                node_landmark_y = data_batch.y
                node_landmark_valid = data_batch.valid_labels
                pix2mm_x = data_batch.pix2mm_x
                pix2mm_y = data_batch.pix2mm_y

                node_coords = None
                node_coord_y = None
                if self.use_coordinate_graph:
                    node_coords = data_batch.node_coords
                    node_coord_y = data_batch.node_coord_y

            # Get frame embeddings
            x = self.model['embedder'](input_frames)

            # put transformed data back into list
            if type(data_batch) == list:
                for idx in range(0, len(data_batch)):
                    data_batch[idx].x = x[idx].unsqueeze(0)

                # Get node predictions
                node_landmark_preds, node_coord_preds = self.model['landmark'](data_batch)

            else:
                node_landmark_preds, node_coord_preds = self.model['landmark'](x=x,
                                                                               node_coords=node_coords,
                                                                               edge_index=data_batch.edge_index,
                                                                               batch_idx=data_batch.batch,
                                                                               node_type=data_batch.node_type)

            # Compute loss
            losses = self.compute_loss(node_landmark_preds,
                                       node_landmark_y,
                                       node_coord_preds,
                                       node_coord_y,
                                       node_landmark_valid)
            if type(losses) == dict:
                loss = sum(losses.values())
            elif type(losses) == float:
                loss = losses
            else:
                raise(f"invalid variable type {type(losses)} for computed losses")

            # Backward propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                # Add to losses
                if type(data_batch) == list:
                    batch_size = len(data_batch)
                else:
                    batch_size = data_batch.batch[-1].item() + 1
                self.loss_meter.update(loss.item(), batch_size)

                # update evaluators
                self.update_evaluators(node_landmark_preds,
                                       node_landmark_y,
                                       node_coord_preds,
                                       node_coord_y,
                                       pix2mm_x,
                                       pix2mm_y,
                                       node_landmark_valid)

                # update tqdm progress bar
                self.set_tqdm_description(iterator, 'train', epoch, loss.item())

                if self.train_config['use_wandb']:
                    step = (epoch*epoch_steps + i)*batch_size
                    self.log_wandb(losses, {"step": step}, mode='batch_train')
                    # plot the heatmaps
                    if num_steps % self.wandb_log_steps == 0:
                        self.log_heatmap_wandb({"step": step},
                                               input_frames,
                                               node_landmark_preds,
                                               node_landmark_y,
                                               node_coord_preds,
                                               pix2mm_x,
                                               pix2mm_y,
                                               mode='batch_train')

                # Save a checkpoint
                num_steps += batch_size
                if num_steps % checkpoint_step == 0:
                    self.checkpointer.save(epoch, num_steps)

        torch.cuda.empty_cache()
        return num_steps

    def evaluate(self, data_type='val'):
        start_epoch, num_steps = 0, 0

        self._build(mode='test')

        self.logger.info('Evaluating the model')

        util.reset_evaluators(self.evaluators)
        self.loss_meter.reset()

        # Evaluate
        train_start = time.time()
        self._evaluate_once(0, num_steps, data_type=data_type, save_output=True)
        validation_time = time.time() - train_start
        # print a summary of the validation epoch
        self.log_summary("Validation", 0, validation_time)
        self.log_wandb({'loss_total': self.loss_meter.avg}, {"epoch": 0}, mode='epoch/valid')

    def _evaluate_once(self, epoch, num_steps, data_type='val', save_output=False):

        if save_output: # for generating a csv output of model's prediction on dataset
            prediction_df = pd.DataFrame()

        dataloader = self.dataloaders[data_type]

        for model_name in self.model.keys():
            self.model[model_name].eval()

        epoch_steps = len(dataloader)
        data_iter = iter(dataloader)
        iterator = tqdm(range(len(dataloader)), dynamic_ncols=True)
        for i in iterator:
            data_batch = next(data_iter)
            with torch.no_grad():

                # Process each sample in the batch
                if type(data_batch) == list:

                    for indx in range(len(data_batch)):
                        data_batch[indx].to(self.device)

                    input_frames = torch.cat([graph.x for graph in data_batch])
                    node_landmark_y = torch.cat([graph.y for graph in data_batch])
                    node_landmark_valid = torch.cat([graph.valid_labels for graph in data_batch])
                    pix2mm_x = torch.hstack([graph.pix2mm_x for graph in data_batch])
                    pix2mm_y = torch.hstack([graph.pix2mm_y for graph in data_batch])

                    node_coord_y = None
                    if self.use_coordinate_graph:
                        node_coord_y = torch.cat([graph.node_coord_y for graph in data_batch])

                else:
                    data_batch.to(self.device)
                    input_frames = data_batch.x
                    node_landmark_y = data_batch.y
                    node_landmark_valid = data_batch.valid_labels
                    pix2mm_x = data_batch.pix2mm_x
                    pix2mm_y = data_batch.pix2mm_y

                    node_coords = None
                    node_coord_y = None
                    if self.use_coordinate_graph:
                        node_coords = data_batch.node_coords
                        node_coord_y = data_batch.node_coord_y

                # Get frame embeddings
                x = self.model['embedder'](input_frames)

                # put transformed data back into list
                if type(data_batch) == list:
                    for idx in range(0, len(data_batch)):
                        data_batch[idx].x = x[idx].unsqueeze(0)

                    # Get node predictions
                    node_landmark_preds, node_coord_preds = self.model['landmark'](data_batch)

                else:
                    node_landmark_preds, node_coord_preds = self.model['landmark'](x=x,
                                                                                   node_coords=node_coords,
                                                                                   edge_index=data_batch.edge_index,
                                                                                   batch_idx=data_batch.batch,
                                                                                   node_type=data_batch.node_type)

                # Compute loss
                losses = self.compute_loss(node_landmark_preds,
                                           node_landmark_y,
                                           node_coord_preds,
                                           node_coord_y,
                                           node_landmark_valid)
                if type(losses) == dict:
                    loss = sum(losses.values())
                elif type(losses) == float:
                    loss = losses
                else:
                    raise (f"invalid variable type {type(losses)} for computed losses")

                # Add to losses
                if type(data_batch) == list:
                    batch_size = len(data_batch)
                else:
                    batch_size = data_batch.batch[-1].item() + 1
                self.loss_meter.update(loss.item(), batch_size)

                # update evaluators
                self.update_evaluators(node_landmark_preds,
                                       node_landmark_y,
                                       node_coord_preds,
                                       node_coord_y,
                                       pix2mm_x,
                                       pix2mm_y,
                                       node_landmark_valid)

                # update tqdm progress bar
                self.set_tqdm_description(iterator, 'validation', epoch, loss.item())

                if self.train_config['use_wandb']:
                    step = (epoch*epoch_steps + i)*batch_size
                    self.log_wandb(losses, {"step": step}, mode='batch_valid')
                    # plot the heatmaps
                    if num_steps % self.wandb_log_steps == 0:
                        self.log_heatmap_wandb({"step": step},
                                               input_frames,
                                               node_landmark_preds,
                                               node_landmark_y,
                                               node_coord_preds,
                                               pix2mm_x,
                                               pix2mm_y,
                                               mode='batch_valid')

                # plot the heatmaps
                num_steps += batch_size

                # ##### creating the prediction log table for wandb #TODO update
                if save_output:
                    prediction_df = pd.concat([prediction_df,  self.create_prediction_df(data_batch)], axis=0)

        if save_output:
            # ###### Prediction Table
            if self.train_config['use_wandb']:
                prediction_log_table = wandb.Table(dataframe=prediction_df)
                wandb.log({f"model_output_{data_type}_dataset": prediction_log_table})
            csv_destination = osp.join(osp.dirname(self.model_config['checkpoint_path']),
                                       f'{data_type}_' +
                                       osp.basename(self.model_config['checkpoint_path'])[:-4] +'.csv')
            prediction_df.to_csv(csv_destination)

        torch.cuda.empty_cache()
        return

    def update_evaluators(self,
                          node_landmark_preds,
                          node_landmark_y,
                          node_coord_preds,
                          node_coord_y,
                          pix2mm_x,
                          pix2mm_y,
                          valid):
        """
        update the evaluators with predictions of the current batch. inputs are in cuda
        """
        node_landmark_preds, node_landmark_y, valid = node_landmark_preds.detach().cpu(), \
                                                      node_landmark_y.detach().cpu(), \
                                                      valid.detach().cpu()
        pix2mm_x, pix2mm_y = pix2mm_x.detach().cpu(), pix2mm_y.detach().cpu()

        if self.use_coordinate_graph:
            node_coord_preds, node_coord_y = node_coord_preds.detach().cpu(), node_coord_y.detach().cpu()

        for metric in self.eval_config["standards"]:
            if metric == 'landmarkcoorderror':
                if self.use_coordinate_graph:
                    self.evaluators[metric].update(node_coord_preds, node_coord_y, pix2mm_x, pix2mm_y, valid)
                else:
                    self.evaluators[metric].update(node_landmark_preds, node_landmark_y, pix2mm_x, pix2mm_y, valid)
            else:
                self.evaluators[metric].update(node_landmark_preds, node_landmark_y, valid)

    def set_tqdm_description(self, iterator, mode, epoch, loss):
        standard_name = self.eval_config["standard"]
        standard_value = self.evaluators[standard_name].get_last()
        last_errors = self.evaluators['landmarkcoorderror'].get_last()
        iterator.set_description("[Epoch {}] | {} | Loss: {:.4f} | "
                                 "{}: {:.4f} | "
                                 "[{ivs:.1f}, {lvid_top:.1f}, {lvid_bot:.1f}, {lvpw:.1f}] | "
                                 "[{ivs_w:.1f}, {lvid_w:.1f}, {lvpw_w:.1f}]" .format(epoch,
                                                                                     mode,
                                                                                     loss,
                                                                                     standard_name,
                                                                                     standard_value,
                                                                                     **last_errors),
                                 refresh=True)

    def log_summary(self, mode, epoch, time):
        """
        log summary after a full training or validation epoch
        """
        standard_name = self.eval_config["standard"]
        standard_value = self.evaluators[standard_name].compute()
        errors = self.evaluators['landmarkcoorderror'].compute()
        self.logger.infov(f'{mode} [Epoch {epoch}] with lr: {self.optimizer.param_groups[0]["lr"]:.7} '
                          f'completed in {str(timedelta(seconds=time)):.7} - '
                          f'loss: {self.loss_meter.avg:.4f} - '
                          f'{standard_name}: {standard_value:.2%} - '
                          'errors [IVS, LVID_TOP, LVID_BOT, LVPW] ='
                          "[{ivs:.4f}, {lvid_top:.4f}, {lvid_bot:.4f}, {lvpw:.4f}] | "
                          "[IVS, LVID, LVPW]: "
                          "_MAE_[{ivs_w:.4f}, {lvid_w:.4f}, {lvpw_w:.4f}] "
                          "_MPE_[{ivs_mpe:.4f}, {lvid_mpe:.4f}, {lvpw_mpe:.4f}]" .format(**errors))

    def log_wandb(self, losses, step_metric, mode='batch_train'):

        if not self.train_config['use_wandb']:
            return

        step_name, step_value = step_metric.popitem()
        standard_name = self.eval_config["standard"]
        if "batch" in mode:
            standard_value = self.evaluators[standard_name].get_last()
            errors = self.evaluators['landmarkcoorderror'].get_last()
            log_dict = {f'{mode}/{step_name}': step_value}
        elif "epoch" in mode:
            standard_value = self.evaluators[standard_name].compute()
            errors = self.evaluators['landmarkcoorderror'].compute()
            log_dict = {f'{step_name}': step_value,   # both train and valid x axis are called epoch
                        'lr': self.optimizer.param_groups[0]['lr']}  # record the Learning Rate
        else:
            raise("invalid mode for wandb logging")

        log_dict.update({f'{mode}/{standard_name}': standard_value,
                         f'{mode}/lvid_top_error': errors['lvid_top'],
                         f'{mode}/lvid_bot_error': errors['lvid_bot'],
                         f'{mode}/lvpw_error': errors['lvpw'],
                         f'{mode}/ivs_error': errors['ivs'],
                         f'{mode}/lvid_w_error': errors['lvid_w'],
                         f'{mode}/lvpw_w_error': errors['lvpw_w'],
                         f'{mode}/ivs_w_error': errors['ivs_w'],
                         f'{mode}/lvid_w_mpe': errors['lvid_mpe'],
                         f'{mode}/lvpw_w_mpe': errors['lvpw_mpe'],
                         f'{mode}/ivs_w_mpe': errors['ivs_mpe'],})

        for loss_name, loss in losses.items():
            loss = loss.item() if type(loss) == torch.Tensor else loss
            log_dict.update({f'{mode}/{loss_name}': loss})

        wandb.log(log_dict)

    def log_heatmap_wandb(self, step_metric,
                          x, landmark_preds, landmark_y,
                          coord_preds, pix2mm_x, pix2mm_y,
                          mode='batch_train'):
        step_name, step_value = step_metric.popitem()

        landmark_preds, landmark_y = landmark_preds.detach().cpu(), landmark_y.detach().cpu()
        pix2mm_x, pix2mm_y = pix2mm_x.detach().cpu(), pix2mm_y.detach().cpu()

        if self.use_coordinate_graph:
            coord_preds = coord_preds.detach().cpu()

        fig = self.evaluators['landmarkcoorderror'].get_heatmaps(x.detach().cpu(),
                                                                 landmark_preds, landmark_y,
                                                                 coord_preds, pix2mm_x, pix2mm_y)
        wandb.log({f'{mode}/heatmaps': fig,
                   f'{mode}/{step_name}': step_value})
        plt.close()

    def compute_loss(self, node_landmark_preds, node_landmark_y, node_coord_preds, node_coord_y, valid_labels=None):
        """
        computes and sums all the loss values to be a single number, ready for backpropagation
        """

        losses = dict()

        node_landmark_preds = node_landmark_preds.view(self.train_config['batch_size'], -1, self.num_output_channels)
        node_landmark_y = node_landmark_y.view(self.train_config['batch_size'], -1, self.num_output_channels)

        for criterion_name in self.criterion.keys():
            if criterion_name == 'coordinate':
                losses[criterion_name] = self.criterion[criterion_name].compute(node_coord_preds, node_coord_y)
            else:
                losses[criterion_name] = self.criterion[criterion_name].compute(node_landmark_preds,
                                                                                node_landmark_y,
                                                                                valid_labels)

        return losses

    def create_prediction_df(self, data_batch):
        """
        creates a pandas dataframe of predictions and labels. suitable for logging on wandb or in .csv
        :return:
        a pandas df with columns: 'db_idx', patientID, landmark coordinates and widths (both gt and predictions)
        """
        # data headers and identifiers
        if type(data_batch) == list:
            # db_idx = [graph.db_indx.item() for graph in data_batch]
            # PatientID = [graph.PatientID.item() for graph in data_batch]
            # StudyDate = [graph.StudyDate.item() for graph in data_batch]
            # SIUID = [graph.SIUID for graph in data_batch]  # no "tolist" is needed
            # file_name = [graph.file_name for graph in data_batch]  # no "tolist" is needed
            # LV_Mass = [graph.LV_Mass.item() for graph in data_batch]
            # BSA = [graph.BSA.item() for graph in data_batch]
            pix2mm_x = torch.hstack([graph.pix2mm_x for graph in data_batch]).tolist()
            pix2mm_y = torch.hstack([graph.pix2mm_y for graph in data_batch]).tolist()
        else:
            # db_idx = data_batch.db_indx.tolist()
            # PatientID = data_batch.PatientID.tolist()
            # StudyDate = data_batch.StudyDate.tolist()
            # SIUID = data_batch.SIUID  # no "tolist" is needed
            # file_name = data_batch.file_name   # no "tolist" is needed
            # LV_Mass = data_batch.LV_Mass.tolist()
            # BSA = data_batch.BSA.tolist()
            pix2mm_x = data_batch.pix2mm_x.tolist()
            pix2mm_y = data_batch.pix2mm_y.tolist()

        pred_log_data = {'pix2mm_x': pix2mm_x, 'pix2mm_y': pix2mm_y}

        # gt and predictions of landmark coordinates and widths
        detailed_performance = self.evaluators['landmarkcoorderror'].get_predictions()
        # gt and predicted coordinates
        pred_log_data.update({key: value.tolist() for key, value in detailed_performance['coordinates'].items()})
        # gt and predicted lengths
        pred_log_data.update({key: value.tolist() for key, value in detailed_performance['widths'].items()})

        return pd.DataFrame(pred_log_data)
