import logging

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm
import os

class Trainer(object):

    def __init__(self, args, model, wandb, save_dir,logger):
        self.args = args
        self.model = model
        self.logger = logger
        self.wandb = wandb 
        self.lr = args.lr
        self.learner = args.learner
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.eval_step = min(args.eval_step, self.epochs)
        self.save_step = min(args.save_step, self.epochs)
        self.device = args.device
        self.device = torch.device(self.device)
        self.ckpt_dir = save_dir

        self.best_loss = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.optimizer = self._build_optimizer()
        self.model = self.model.to(self.device)

    def _build_optimizer(self):

        params = self.model.parameters()
        learner =  self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def _train_epoch(self, train_data, epoch_idx):

        self.model.train()

        total_loss = []
        total_recon_loss =[]
        total_cap_loss = []
        total_rq_loss = []

        for batch_idx, data in enumerate(train_data):
            
            img = data[0].to(self.device) if self.args.use_cap else data.to(self.device)
            cap = data[1].to(self.device) if self.args.use_cap else None

            self.optimizer.zero_grad()
            out, rq_loss, indices, encoder_out = self.model(img)
            loss, loss_recon, loss_cap = self.model.compute_loss(out, encoder_out,rq_loss, xs=img,cap=cap)
            self._check_nan(loss)
            loss.backward()
            self.optimizer.step()

            total_loss.append(loss.item())
            total_recon_loss.append(loss_recon.item())
            total_cap_loss.append(loss_cap.item())
            total_rq_loss.append(rq_loss.item())

        return np.mean(total_loss),np.mean(total_recon_loss),np.mean(total_cap_loss),np.mean(total_rq_loss)


    @torch.no_grad()
    def _valid_epoch(self, valid_data):

        self.model.eval()

        loss_total = []
        loss_recon_total = []
        loss_cap_total = []
        loss_rq_total = []
        for batch_idx, data in enumerate(valid_data):
            #data = data.to(self.device)
            img = data[0].to(self.device) if self.args.use_cap else data.to(self.device)
            cap = data[1].to(self.device) if self.args.use_cap else None
            with torch.no_grad():
                out, rq_loss, indices,encoder_out = self.model(img)
                loss, loss_recon,loss_cap = self.model.compute_loss(out,encoder_out, rq_loss, xs=img,cap=cap)
                loss_total.append(loss.item())
                loss_recon_total.append(loss_recon.item())
                loss_cap_total.append(loss_cap.item())
                loss_rq_total.append(rq_loss.item())
        loss_mean = np.mean(loss_total)
        loss_recon_mean = np.mean(loss_recon_total)
        loss_cap_mean = np.mean(loss_cap_total)
        loss_rq_mean = np.mean(loss_rq_total)

        return loss_mean, loss_recon_mean,loss_cap_mean,loss_rq_mean

    def _save_checkpoint(self, epoch, loss, ckpt_file=None):

        ckpt_path = os.path.join(self.ckpt_dir,ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, 'epoch_%d_val_loss_%.4f_model.pth' % (epoch + 1, loss))
        state = {
            "args": self.args,
            "epoch": epoch + 1,
            "best_loss": self.best_loss,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)

        self.logger.info(
            "Saving current" + f": {ckpt_path}"
        )
        

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, recon_loss):
        train_loss_output = (
            "epoch %d training"
            + " ["
            + "time"
            + ": %.2fs, "
        ) % (epoch_idx + 1, e_time - s_time)
        train_loss_output += "train loss" + ": %.4f" % loss
        train_loss_output +=", "
        train_loss_output += "reconstruction loss" + ": %.4f" % recon_loss
        return train_loss_output + "]"


    def fit(self, train_data,valid_data):

        for epoch_idx in tqdm(range(self.epochs),desc="Training",total=self.epochs):
            # train
            s_time = time()
            train_loss, train_recon_loss,train_cap_loss,train_rq_loss = self._train_epoch(train_data, epoch_idx)
            e_time = time()

            self.wandb.log({"train_loss":train_loss,"train_recon_loss":train_recon_loss,"train_cap_loss":train_cap_loss,"train_rq_loss":train_rq_loss})
            self.logger.info(
                self._generate_train_loss_output(epoch_idx, s_time, e_time, train_loss,train_recon_loss)
            )
            # eval
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                val_loss,val_loss_recon,val_loss_cap,val_rq_loss = self._valid_epoch(valid_data)
                # if (epoch_idx + 1) % self.save_step == 0:
                #     self._save_checkpoint(epoch_idx, val_loss)

                valid_end_time = time()
                valid_score_output = (
                    "epoch %d evaluating"
                    + " ["
                    + "time"
                    + ": %.2fs, "
                    + "loss"
                    + ": %f"
                    + "loss_recon"
                    + ": %f]"
                ) % (epoch_idx + 1, valid_end_time - valid_start_time, val_loss,val_loss_recon)

                self.wandb.log({"val_loss":val_loss,"val_recon_loss":val_loss_recon,"val_cap_loss":val_loss_cap,"val_rq_loss":val_rq_loss})
                self.logger.info(valid_score_output)
            
            # save 
            if (epoch_idx + 1) % self.save_step == 0:
                    self._save_checkpoint(epoch_idx, train_loss)

        return self.best_loss




