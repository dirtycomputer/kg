import torch
from torch import optim
from tqdm import tqdm
from sklearn.metrics import classification_report
from transformers.optimization import get_linear_schedule_with_warmup
from .metrics import eval_result, compute_calibration_metrics
import os
import json
import logging

logger = logging.getLogger(__name__)


class Trainer(object):
    """训练器（集成新 loss 组合 + 分项日志）"""

    def __init__(self, train_data=None, dev_data=None, test_data=None,
                 re_dict=None, model=None, args=None, logger=None, writer=None):
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.re_dict = re_dict
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
        self.writer = writer
        self.refresh_step = 2

        self.best_dev_epoch = None
        self.best_test_epoch = None
        self.best_test_metric = 0
        self.test_acc = 0
        self.best_dev_metric = 0
        self.dev_acc = 0

        self.optimizer = None
        self.step = 0
        self.args = args

        if self.train_data is not None:
            self.train_num_steps = len(self.train_data) * args.num_epochs
            self._setup_optimizer()

    def _setup_optimizer(self):
        """配置优化器和学习率调度器"""
        params = {'lr': self.args.lr, 'weight_decay': 1e-2, 'params': []}
        for name, param in self.model.named_parameters():
            params['params'].append(param)

        self.optimizer = optim.AdamW([params], lr=self.args.lr)
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=int(self.args.warmup_ratio * self.train_num_steps),
            num_training_steps=self.train_num_steps
        )
        self.model.to(self.args.device)

    def train(self):
        self.step = 0
        self.model.train()
        self.logger.info("***** Running training *****")
        self.logger.info("  Num instance = %d", len(self.train_data) * self.args.batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Learning rate = {}".format(self.args.lr))
        self.logger.info("  Evaluate begin = %d", self.args.eval_begin_epoch)

        if self.args.load_path is not None:
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))

        with tqdm(total=self.train_num_steps, postfix='loss:{0:<6.5f}', leave=False,
                  dynamic_ncols=True, initial=self.step) as pbar:
            self.pbar = pbar
            avg_loss = 0
            loss_accum = {'ce': 0, 'seq_vib': 0, 'entity_vib': 0, 'gate_contra': 0, 'gate_entropy': 0}

            for epoch in range(1, self.args.num_epochs + 1):
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.args.num_epochs))
                for batch in self.train_data:
                    self.step += 1
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    params, labels = batch

                    # Forward
                    result = self._step(params, labels, mode='train')
                    loss, logits, loss_dict = result

                    avg_loss += loss.detach().cpu().item()
                    for k in loss_accum:
                        if k in loss_dict:
                            loss_accum[k] += loss_dict[k].detach().cpu().item()

                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    if self.step % self.refresh_step == 0:
                        avg_loss = float(avg_loss) / self.refresh_step
                        print_parts = ["loss:{:<6.5f}".format(avg_loss)]
                        for k in loss_accum:
                            val = loss_accum[k] / self.refresh_step
                            print_parts.append("{}:{:<6.4f}".format(k, val))
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str(" ".join(print_parts))

                        if self.writer is not None:
                            self.writer.add_scalar('train/loss_total', avg_loss, self.step)
                            for k in loss_accum:
                                self.writer.add_scalar(f'train/loss_{k}', loss_accum[k] / self.refresh_step, self.step)

                        avg_loss = 0
                        loss_accum = {k: 0 for k in loss_accum}

                if epoch >= self.args.eval_begin_epoch:
                    self.evaluate(epoch)
                    self.test(epoch)

            pbar.close()
            self.pbar = None
            self.logger.info(
                "Best dev f1: {} at epoch {}, acc = {}".format(
                    self.best_dev_metric, self.best_dev_epoch, self.dev_acc))
            self.logger.info(
                "Best test f1: {} at epoch {}, acc = {}".format(
                    self.best_test_metric, self.best_test_epoch, self.test_acc))

    def evaluate(self, epoch):
        self.model.eval()
        self.logger.info("***** Running evaluate *****")
        true_labels, pred_labels = [], []
        all_logits = []

        with torch.no_grad():
            with tqdm(total=len(self.dev_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Evaluating")
                total_loss = 0
                for batch in self.dev_data:
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    params, labels = batch
                    loss, logits, _ = self._step(params, labels, mode='eval')
                    total_loss += loss.detach().cpu().item()

                    preds = logits.argmax(-1)
                    true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    pred_labels.extend(preds.view(-1).detach().cpu().tolist())
                    all_logits.append(logits.detach().cpu())
                    pbar.update()
                pbar.close()

                sk_result = classification_report(
                    y_true=true_labels, y_pred=pred_labels,
                    labels=list(self.re_dict.values())[1:],
                    target_names=list(self.re_dict.keys())[1:], digits=4)
                self.logger.info("%s\n", sk_result)

                result = eval_result(true_labels, pred_labels, self.re_dict, self.logger)
                acc = round(result['acc'] * 100, 4)
                micro_f1 = round(result['micro_f1'] * 100, 4)

                # 校准指标
                all_logits_t = torch.cat(all_logits, dim=0)
                all_labels_t = torch.tensor(true_labels)
                calib = compute_calibration_metrics(all_logits_t, all_labels_t)
                self.logger.info("Calibration: ECE={:.4f}, Brier={:.4f}, NLL={:.4f}".format(
                    calib['ece'], calib['brier'], calib['nll']))

                self.logger.info(
                    "Epoch {}/{}, best dev f1: {}, best epoch: {}, current f1: {}, acc: {}".format(
                        epoch, self.args.num_epochs, self.best_dev_metric,
                        self.best_dev_epoch, micro_f1, acc))

                if micro_f1 >= self.best_dev_metric:
                    self.logger.info("Get better dev performance at epoch {}".format(epoch))
                    self.best_dev_epoch = epoch
                    self.best_dev_metric = micro_f1
                    self.dev_acc = acc

        self.model.train()

    def test(self, epoch):
        self.model.eval()
        self.logger.info("\n***** Running testing *****")
        true_labels, pred_labels = [], []
        all_logits = []

        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Testing")
                total_loss = 0
                for batch in self.test_data:
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    params, labels = batch
                    loss, logits, _ = self._step(params, labels, mode='eval')
                    total_loss += loss.detach().cpu().item()

                    preds = logits.argmax(-1)
                    true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    pred_labels.extend(preds.view(-1).detach().cpu().tolist())
                    all_logits.append(logits.detach().cpu())
                    pbar.update()
                pbar.close()

                sk_result = classification_report(
                    y_true=true_labels, y_pred=pred_labels,
                    labels=list(self.re_dict.values())[1:],
                    target_names=list(self.re_dict.keys())[1:], digits=4)
                self.logger.info("%s\n", sk_result)

                result = eval_result(true_labels, pred_labels, self.re_dict, self.logger)
                acc = round(result['acc'] * 100, 4)
                micro_f1 = round(result['micro_f1'] * 100, 4)

                # 校准指标
                all_logits_t = torch.cat(all_logits, dim=0)
                all_labels_t = torch.tensor(true_labels)
                calib = compute_calibration_metrics(all_logits_t, all_labels_t)
                self.logger.info("Calibration: ECE={:.4f}, Brier={:.4f}, NLL={:.4f}".format(
                    calib['ece'], calib['brier'], calib['nll']))

                self.logger.info(
                    "Epoch {}/{}, best test f1: {}, best epoch: {}, current f1: {}, acc: {}".format(
                        epoch, self.args.num_epochs, self.best_test_metric,
                        self.best_test_epoch, micro_f1, acc))

                if micro_f1 >= self.best_test_metric:
                    self.logger.info("Get better test performance at epoch {}".format(epoch))
                    self.best_test_epoch = epoch
                    self.best_test_metric = micro_f1
                    self.test_acc = acc
                    if self.args.save_path is not None:
                        torch.save(self.model.state_dict(), os.path.join(self.args.save_path, "best_model.pth"))
                        self.logger.info("Save best model at {}".format(self.args.save_path))

        self.model.train()

    def _step(self, params, labels, mode='train'):
        """单步前向传播"""
        # 注入 mode 参数
        params['mode'] = mode
        outputs = self.model(**params)
        if isinstance(outputs, tuple) and len(outputs) == 3:
            loss, logits, loss_dict = outputs
        else:
            # fallback: 兼容只返回 (loss, logits) 的情况
            loss, logits = outputs[0], outputs[1]
            loss_dict = {'ce': loss}
        return loss, logits, loss_dict
