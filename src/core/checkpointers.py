import os
import time

import numpy
import torch
from collections import OrderedDict


class CustomCheckpointer(object):
    def __init__(self, checkpoint_dir, logger, model,
                 optimizer, scheduler, eval_standard='accuracy', best_mode='max'):
        self.logger = logger
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.eval_standard = eval_standard
        self.reset(best_mode)
        self.logger.infov('Checkpointer is built.')

    def reset(self, best_mode):
        if best_mode == 'max':
            self.best_eval_metric = 0.
            self.last_eval_metric = 0.
        elif best_mode == 'min':
            self.best_eval_metric = numpy.Inf
            self.last_eval_metric = numpy.Inf

    def save(self, epoch, num_steps, eval_metric=None, best_mode='max'):
        model_params = {'epoch': epoch, 'num_steps': num_steps}

        if torch.cuda.device_count() > 1:
            model_params['embedder_state_dict'] = self.model['embedder'].module.state_dict()
            model_params['landmark_state_dict'] = self.model['landmark'].module.state_dict()
        else:
            model_params['embedder_state_dict'] = self.model['embedder'].state_dict()
            model_params['landmark_state_dict'] = self.model['landmark'].state_dict()

        model_params['optimizer_state_dict'] = self.optimizer.state_dict()

        # Save the given checkpoint in train time and last/best checkpoints in eval time.
        if eval_metric is None:
            checkpoint_path = os.path.join(
                self.checkpoint_dir, 'checkpoint' + '_' + str(epoch) + '_' + str(num_steps) + '.pth')
            torch.save(model_params, checkpoint_path)
        else:
            # Save the last checkpoint
            self.last_eval_metric = eval_metric
            last_checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint_last.pth')
            torch.save(model_params, last_checkpoint_path)

            # Update the last checkpoint path
            self._record_last_checkpoint_path()

            # Save the best checkpoint if current eval metric is better than the best one
            if best_mode == 'max' and eval_metric <= self.best_eval_metric:
                return
            if best_mode == 'min' and eval_metric >= self.best_eval_metric:
                return
            self.best_eval_metric = eval_metric

            best_checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint_best.pth')
            torch.save(model_params, best_checkpoint_path)

            # Update the checkpoint record
            best_checkpoint_info = {'epoch': epoch, 'num_steps': num_steps}
            best_checkpoint_info.update({'accuracy': eval_metric})
            self._record_best_checkpoint(best_checkpoint_info)

        self.logger.info(
            'A checkpoint is saved for epoch={}, steps={}.'.format(
                epoch, num_steps))

    def load(self, mode, checkpoint_path=None, use_latest=False):
        if mode == 'train':
            if self._has_checkpoint() and use_latest:
                checkpoint_path = self._get_checkpoint_path()
            if not checkpoint_path:
                self.logger.info("No checkpoint found. Initializing model from scratch.")
                return {}
        else:
            if not checkpoint_path:
                while not self._has_checkpoint():
                    self.logger.warn('No checkpoint available. Wait for 60 seconds.')
                    time.sleep(60)
                checkpoint_path = self._get_checkpoint_path()

        self.logger.info("Loading checkpoint from {}".format(checkpoint_path))
        checkpoint_dict = self._load_checkpoint(checkpoint_path)

        self.model['embedder'].load_state_dict(
            checkpoint_dict.pop('embedder_state_dict'), strict=True)

        self.model['landmark'].load_state_dict(
            checkpoint_dict.pop('landmark_state_dict'), strict=True)

        if 'optimizer_state_dict' in checkpoint_dict and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(checkpoint_path))
            self.optimizer.load_state_dict(checkpoint_dict.pop('optimizer_state_dict'))
        if 'scheduler_state_dict' in checkpoint_dict and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(checkpoint_path))
            self.scheduler.load_state_dict(checkpoint_dict.pop('scheduler_state_dict'))

        return checkpoint_dict

    def _freeze(self):
        for param in self.model.layers.parameters():
            param.requires_grad = False

    def _has_checkpoint(self):
        record_path = os.path.join(self.checkpoint_dir, "last_checkpoint")
        return os.path.exists(record_path)

    def _get_checkpoint_path(self):
        record_path = os.path.join(self.checkpoint_dir, "last_checkpoint")
        try:
            with open(record_path, "r") as f:
                last_saved = f.readlines()
                last_saved = last_saved.strip()
        except IOError:
            self.logger.warn('If last_checkpoint file doesn not exist, maybe because \
                              it has just been deleted by a separate process.')
            last_saved = ''
        return last_saved

    def _record_last_checkpoint_path(self):
        record_path = os.path.join(self.checkpoint_dir, 'last_checkpoint')
        with open(record_path, 'w') as f:
            f.write(str(record_path))

    def _record_best_checkpoint(self, best_checkpoint_info):
        record_path = os.path.join(self.checkpoint_dir, 'best_checkpoint')
        with open(record_path, 'w') as f:
            f.write(str(best_checkpoint_info))

    @staticmethod
    def _load_checkpoint(checkpoint_path):
        checkpoint_dict = torch.load(checkpoint_path)
        if torch.cuda.device_count() > 1:
            for model_name in ['landmark', 'embedder']:
                checkpoint = checkpoint_dict[model_name + '_state_dict']
                checkpoint = OrderedDict([('module.'+ k, v) for k, v in checkpoint.items()])
                checkpoint_dict[model_name + '_state_dict'] = checkpoint

        checkpoint_dict['checkpoint_path'] = checkpoint_path
        return checkpoint_dict
