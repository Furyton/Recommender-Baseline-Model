import json
from abc import *
from pathlib import Path

import time

import torch
from config import *
from loggers import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from trainers.BaseTrainer import AbstractBaseTrainer
from trainers.loss import CELoss
from trainers.utils import recalls_ndcgs_and_mrr_for_ks
from utils import AverageMeterSet


class NormalTrainer(AbstractBaseTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root, prefix='models'):
        super().__init__(args, model, export_root)
        self.enable_mentor = args.enable_mentor
        self.num_epochs = args.num_epochs
        self.prefix = prefix
        self.model : torch.nn.Module

        self.accum_iter = 0
        self._load_state()

        self.loss = CELoss(self.args)

        self.optimizer = self._create_optimizer()

        self._load_opt_state()
        
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.decay_step, gamma=args.gamma)


        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric

        self.writer, self.train_loggers, self.val_loggers = self._create_loggers()
        # self.add_extra_loggers()
        self.iter_per_epoch = len(self.train_loader) * self.batch_size
        self.tot_iter = self.num_epochs * self.iter_per_epoch

        print('{} iter per epoch'.format(self.iter_per_epoch))

        self.logger_service = LoggerService(
            self.train_loggers, self.val_loggers)
        self.log_period_as_iter = args.log_period_as_iter

        self.enable_neg_sample = args.test_negative_sample_size != 0
        self.num_items = args.num_items

    @classmethod
    def code(cls):
        return 'normal'

    def _load_state(self):
        if self.prefix == 'models':
            if self.args.model_state_path:
                print("loading model\'s parameters")

                checkpoint = torch.load(
                    Path(self.args.model_state_path), map_location=torch.device(self.device))
                print("checkpoint epoch number: ", checkpoint['epoch'])
                self.model.load_state_dict(checkpoint[STATE_DICT_KEY])
                self.accum_iter = checkpoint[ACCUM_ITER_DICT_KEY]
        else:
            if self.args.mentor_state_path:
                print("loading model\'s parameters")

                checkpoint = torch.load(
                    Path(self.args.mentor_state_path), map_location=torch.device(self.device))
                print("checkpoint epoch number: ", checkpoint['epoch'])
                self.model.load_state_dict(checkpoint[STATE_DICT_KEY])
                self.accum_iter = checkpoint[ACCUM_ITER_DICT_KEY]

    def _load_opt_state(self):
        if self.prefix == 'models':
            if self.args.model_state_path:
                print("loading optimizer\'s parameters")

                checkpoint = torch.load(
                    Path(self.args.model_state_path), map_location=torch.device(self.device))
                print("checkpoint epoch number: ", checkpoint['epoch'])
                self.optimizer.load_state_dict(
                    (checkpoint[OPTIMIZER_STATE_DICT_KEY]))
        else:
            if self.args.mentor_state_path:
                print("loading optimizer\'s parameters")

                checkpoint = torch.load(
                    Path(self.args.mentor_state_path), map_location=torch.device(self.device))
                print("checkpoint epoch number: ", checkpoint['epoch'])
                self.optimizer.load_state_dict(
                    (checkpoint[OPTIMIZER_STATE_DICT_KEY]))

    def close_training(self):
        pass

    def train(self):
        # self.accum_iter = 0

        # print("\n[debug] model initial params\n")
        # for name, param in self.model.named_parameters():
        #     print (f"name {name}, param.data {param.data}")
        
        self.validate(0)
        for epoch in range(self.num_epochs):
            print("epoch: ", epoch)

            t = time.time()

            self.train_one_epoch(epoch)
            self.validate(epoch)

            print("duration: ", time.time() - t, 's')

            self.lr_scheduler.step()

        self.logger_service.complete({
            'state_dict': (self._create_state_dict()),
        })
        self.writer.close()
        self.close_training()

    def calculate_loss(self, batch):
        batch = [x.to(self.device) for x in batch]
        # seqs, labels, rating = batch

        return self.loss.compute(self.model(batch), batch)

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def train_one_epoch(self, epoch):
        self.model.train()

        average_meter_set = AverageMeterSet()
        # tqdm_dataloader = tqdm(self.train_loader)

        iterator = self.train_loader if not self.args.show_process_bar else tqdm(self.train_loader)

        tot_loss = 0.
        tot_batch = 0

        for batch_idx, batch in enumerate(iterator):
            # batch_size = batch[0].size(0)

            self.optimizer.zero_grad()
            loss = self.calculate_loss(batch)

            tot_loss += loss.item()

            tot_batch += 1

            loss.backward()

            self.optimizer.step()

            average_meter_set.update('loss', loss.item())

            if self.args.show_process_bar:
                iterator.set_description('Epoch {}, loss {:.3f} '.format(
                    epoch + 1, average_meter_set['loss'].avg))

            self.accum_iter += self.batch_size

            if self._needs_to_log(self.accum_iter):

                if self.args.show_process_bar:
                    iterator.set_description('Logging to Tensorboard')

                self.writer.add_scalar(
                    "learning_rate", self.get_lr(), self.accum_iter)

                log_data = {
                    'state_dict': (self._create_state_dict()),
                    'epoch': epoch,
                    'accum_iter': self.accum_iter,
                }
                log_data.update(average_meter_set.averages())
                # self.log_extra_train_info(log_data)
                self.logger_service.log_train(log_data)

        print(tot_loss / tot_batch)

    def calculate_metrics(self, batch):
        batch = [x.to(self.device) for x in batch]

        if self.enable_neg_sample:
            raise NotImplementedError("codes for evaluating with negative candidates has bug")
            scores = self.model.predict(batch)
        else:
            # seqs, answer, ratings, ... = batch
            seqs = batch[0]
            answer = batch[1]
            ratings = batch[2]

            batch_size = len(seqs)
            labels = torch.zeros(batch_size, self.num_items + 1, device=self.device)
            scores = self.model.full_sort_predict(batch)

            row = []
            col = []

            for i in range(batch_size):
                seq = list(set(seqs[i].tolist()) | set(answer[i].tolist()))
                seq.remove(answer[i][0].item())
                if self.num_items + 1 in seq:
                    seq.remove(self.num_items + 1)
                row += [i] * len(seq)
                col += seq
                labels[i][answer[i]] = 1
            scores[row, col] = -1e9

        metrics = recalls_ndcgs_and_mrr_for_ks(scores, labels, self.metric_ks, ratings)
        return metrics

    def validate(self, epoch):
        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            iterator = self.val_loader if not self.args.show_process_bar else tqdm(
                self.val_loader)

            for batch_idx, batch in enumerate(iterator):
                # batch = [x.to(self.device) for x in batch]
                metrics = self.calculate_metrics(batch)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)

                if self.args.show_process_bar:
                    description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] + \
                                          ['Recall@%d' % k for k in self.metric_ks[:3]] + \
                                          ['MRR@%d' %
                                              k for k in self.metric_ks[:3]]
                    description = 'Val: ' + \
                        ', '.join(s + ' {:.3f}' for s in description_metrics)
                    description = description.replace('NDCG', 'N').replace(
                        'Recall', 'R').replace('MRR', 'M')
                    description = description.format(
                        *(average_meter_set[k].avg for k in description_metrics))
                    iterator.set_description(description)

            log_data = {
                'state_dict': (self._create_state_dict()),
                'epoch': epoch,
                'accum_iter': self.accum_iter,
            }
            log_data.update(average_meter_set.averages())
            self.logger_service.log_val(log_data)
            average_metrics = average_meter_set.averages()
            print(average_metrics)

    def test(self):
        print('Test best model with test set!')

        if self.args.test_model_path is not None:
            best_model = torch.load(Path(self.args.test_model_path), map_location=torch.device(self.args.device)).get(
                'model_state_dict')
        else:
            best_model = torch.load(os.path.join(self.export_root, self.prefix, 'best_acc_model.pth')).get('model_state_dict')

        self.model.load_state_dict(best_model)

        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            iterator = self.test_loader if not self.args.show_process_bar else tqdm(
                self.test_loader)

            for batch_idx, batch in enumerate(iterator):

                metrics = self.calculate_metrics(batch)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)

                if self.args.show_process_bar:
                    description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] + \
                                          ['Recall@%d' % k for k in self.metric_ks[:3]] + \
                                          ['MRR@%d' %
                                              k for k in self.metric_ks[:3]]
                    description = 'Val: ' + \
                        ', '.join(s + ' {:.3f}' for s in description_metrics)
                    description = description.replace('NDCG', 'N').replace(
                        'Recall', 'R').replace('MRR', 'M')
                    description = description.format(
                        *(average_meter_set[k].avg for k in description_metrics))
                    iterator.set_description(description)

        average_metrics = average_meter_set.averages()
        print(average_metrics)

        if not os.path.exists(os.path.join(self.export_root, self.prefix + '_logs')):
            os.mkdir(os.path.join(self.export_root, self.prefix + '_logs'))

        with open(os.path.join(self.export_root, self.prefix + '_logs', 'test_metrics.json'), 'w') as f:
            json.dump(average_metrics, f)

    def _create_loggers(self):
        root = Path(self.export_root)
        writer = SummaryWriter(root.joinpath('logs', 'tb_vis_' + self.prefix))
        model_checkpoint = root.joinpath(self.prefix)

        train_loggers = [
            MetricGraphPrinter(writer, key='epoch',
                               graph_name='Epoch', group_name='Train'),
            MetricGraphPrinter(writer, key='loss',
                               graph_name='Loss', group_name='Train'),
        ]

        val_loggers = []
        for k in self.metric_ks:
            val_loggers.append(
                MetricGraphPrinter(writer, key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Validation'))
            val_loggers.append(
                MetricGraphPrinter(writer, key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Validation'))
            val_loggers.append(
                MetricGraphPrinter(
                    writer, key='NDCG*@%d' % k, graph_name='NDCG*@%d' % k, group_name='Validation')
            )
            val_loggers.append(
                MetricGraphPrinter(writer, key='Recall*@%d' % k, graph_name='Recall*@%d' % k, group_name='Validation'))
        val_loggers.append(RecentModelLogger(model_checkpoint))
        val_loggers.append(BestModelLogger(
            model_checkpoint, metric_key=self.best_metric))
        return writer, train_loggers, val_loggers

    def _needs_to_log(self, accum_iter):
        return accum_iter % self.log_period_as_iter < self.args.train_batch_size and accum_iter != 0

    def _create_optimizer(self):
        args = self.args
        if args.optimizer.lower() == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                momentum=args.momentum)
        else:
            raise ValueError
