"""
Train Classes used to train models in main.py
"""

from utils.metrics import *
from sklearn.metrics import *
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from time import time
import numpy as np
import gc
import os
import logging
from utils.torch_utils import move_batch_to_gpu
from utils.utility_funcs import format_metric

from evals import *


def get_filter_mask(filter_num):
    return np.random.choice([0, 1], size=(filter_num,))


def get_masked_disc(disc_dict, labels, mask):
    if np.sum(mask) == 0:
        return []
    masked_disc_label = [
        (disc_dict[i + 1], labels[:, i]) for i, val in enumerate(mask) if val != 0
    ]
    return masked_disc_label


def check(model, out_dict, l2_weight):
    """
    Check intermediate results
    :param model: model obj
    :param out_dict: output dictionary
    :return: None
    """
    check = out_dict
    logging.info(os.linesep)
    for i, t in enumerate(check["check"]):
        d = np.array(t[1].detach().cpu())
        logging.info(
            os.linesep.join(
                [t[0] + "\t" + str(d.shape), np.array2string(d, threshold=20)]
            )
            + os.linesep
        )

    loss, l2 = check["loss"], model.l2()
    l2 = l2 * l2_weight
    l2 = l2.detach()
    logging.info("loss = %.4f, l2 = %.4f" % (loss, l2))
    if not (np.absolute(loss) * 0.005 < l2 < np.absolute(loss) * 0.1):
        logging.warning("l2 inappropriate: loss = %.4f, l2 = %.4f" % (loss, l2))

    # for discriminator
    disc_score_dict = out_dict["d_score"]
    for feature in disc_score_dict:
        logging.info("{} AUC = {:.4f}".format(feature, disc_score_dict[feature]))


def check_disc(out_dict):
    check = out_dict
    logging.info(os.linesep)
    for i, t in enumerate(check["check"]):
        d = np.array(t[1].detach().cpu())
        logging.info(
            os.linesep.join(
                [t[0] + "\t" + str(d.shape), np.array2string(d, threshold=20)]
            )
            + os.linesep
        )

    loss_dict = check["loss"]
    for disc_name, disc_loss in loss_dict.items():
        logging.info("%s loss = %.4f" % (disc_name, disc_loss))

    # for discriminator
    if "d_score" in out_dict:
        disc_score_dict = out_dict["d_score"]
        for feature in disc_score_dict:
            logging.info("{} AUC = {:.4f}".format(feature, disc_score_dict[feature]))


class TrainModel:
    def __init__(
        self,
        feature_info,
        reg_weight,
        no_filter,
        metrics,
        epoch,
        batch_size,
        disc_steps=10,
        num_worker=1,
        check_epoch=1,
        l2_weight=1e-4,
        learning_rate=0.01,
    ):
        self.no_filter = no_filter
        self.reg_weight = reg_weight
        self.epoch = epoch
        self.batch_size = batch_size
        self.disc_steps = disc_steps
        self.num_worker = num_worker
        self.check_epoch = check_epoch
        self.l2_weight = l2_weight
        self.metrics = metrics.lower().split(",")
        self.feature_info = feature_info

        # record train, validation, test results
        self.train_results, self.valid_results, self.test_results = [], [], []
        self.disc_results = []
        self.time = None
        self.lr = learning_rate
        self.time = None

    def _check_time(self, start=False):
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def fit(
        self, model, batches, fair_disc_dict, epoch=-1
    ):  # fit the results for an input set
        """
        Train the model
        :param model: model instance
        :param batches: train data in batches
        :param fair_disc_dict: fairness discriminator dictionary
        :param epoch: epoch number
        :return: return the output of the last round
        """
        gc.collect()
        torch.cuda.empty_cache()

        logging.info("Optimizer: Adam")
        model.optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.l2_weight
        )

        model.train()

        for idx in fair_disc_dict:
            discriminator = fair_disc_dict[idx]
            logging.info("Discriminator Optimizer: Adam")
            discriminator.optimizer = torch.optim.Adam(
                discriminator.parameters(), lr=self.lr, weight_decay=self.l2_weight
            )
            discriminator.train()

        loss_list = list()
        output_dict = dict()
        eval_dict = None
        for batch in tqdm(
            batches,
            leave=False,
            desc="Epoch %5d" % (epoch + 1),
            ncols=100,
            mininterval=1,
        ):
            # get mask functions for filter
            if self.no_filter:
                mask = [0] * model.num_features
                mask = np.asarray(mask)
            else:
                mask = get_filter_mask(model.num_features)

            batch = move_batch_to_gpu(batch)
            model.optimizer.zero_grad()

            labels = batch["features"][: len(batch["features"]) // 2, :]
            if not self.no_filter:
                masked_disc_label = get_masked_disc(fair_disc_dict, labels, mask)
            else:
                masked_disc_label = get_masked_disc(fair_disc_dict, labels, mask + 1)

            # calculate recommendation loss + fair discriminator penalty and back propogate them
            result_dict = model(batch, mask)
            rec_loss = result_dict["loss"]
            vectors = result_dict["u_vectors"]
            vectors = vectors[: len(vectors) // 2, :]

            fair_d_penalty = 0
            if not self.no_filter:
                for fair_disc, label in masked_disc_label:
                    fair_d_penalty += fair_disc(vectors, label)
                fair_d_penalty *= -1
                loss = rec_loss + self.reg_weight * fair_d_penalty
            else:
                loss = rec_loss
            loss.backward()
            model.optimizer.step()

            #update results for recommendation system
            loss_list.append(result_dict["loss"].detach().cpu().data.numpy())
            output_dict["check"] = result_dict["check"]

            # update discriminator
            if not self.no_filter:
                if len(masked_disc_label) != 0:
                    for _ in range(self.disc_steps):
                        for discriminator, label in masked_disc_label:
                            discriminator.optimizer.zero_grad()
                            disc_loss = discriminator(vectors.detach(), label)
                            disc_loss.backward(retain_graph=False)
                            discriminator.optimizer.step()

            # collect discriminator evaluation results
            if eval_dict is None:
                eval_dict = eval_discriminator(
                    model, labels, vectors.detach(), fair_disc_dict, len(mask)
                )

            else:
                batch_eval_dict = eval_discriminator(
                    model, labels, vectors.detach(), fair_disc_dict, len(mask)
                )
                for f_name in eval_dict:
                    new_label = batch_eval_dict[f_name]["label"]
                    current_label = eval_dict[f_name]["label"]
                    eval_dict[f_name]["label"] = torch.cat(
                        (current_label, new_label), dim=0
                    )

                    new_prediction = batch_eval_dict[f_name]["prediction"]
                    current_prediction = eval_dict[f_name]["prediction"]
                    eval_dict[f_name]["prediction"] = torch.cat(
                        (current_prediction, new_prediction), dim=0
                    )

        # generate discriminator evaluation scores
        d_score_dict = {}
        if eval_dict is not None:
            for f_name in eval_dict:
                l = eval_dict[f_name]["label"]
                pred = eval_dict[f_name]["prediction"]
                n_class = eval_dict[f_name]["num_class"]
                d_score_dict[f_name] = disc_eval_method(
                    l, pred, n_class
                )  ######## DISC_EVAL IS CALLED

        output_dict["d_score"] = d_score_dict
        output_dict["loss"] = np.mean(loss_list)
        return output_dict

    def train(self, model, dp_dict, fair_disc_dict, skip_eval=0, fix_one=False):
        """
        Train model
        :param model: model obj
        :param dp_dict: Data processors for train valid and test
        :param skip_eval: number of epochs to skip for evaluations
        :param fair_disc_dict: fairness discriminator dictionary
        :return:
        """
        # prepare data
        train_data = DataLoader(
            dp_dict["train"],
            batch_size=self.batch_size,
            num_workers=self.num_worker,
            shuffle=True,
            collate_fn=dp_dict["train"].collate_fn,
        )
        validation_data = DataLoader(
            dp_dict["valid"],
            batch_size=None,
            num_workers=self.num_worker,
            pin_memory=True,
            collate_fn=dp_dict["test"].collate_fn,
        )
        test_data = DataLoader(
            dp_dict["test"],
            batch_size=None,
            num_workers=self.num_worker,
            pin_memory=True,
            collate_fn=dp_dict["test"].collate_fn,
        )

        self._check_time(start=True)  # start time
        try:
            # per epoch training
            for epoch in range(self.epoch):
                self._check_time()
                output_dict = self.fit(model, train_data, fair_disc_dict, epoch=epoch)
                if self.check_epoch > 0 and (
                    epoch == 1 or epoch % self.check_epoch == 0
                ):
                    check(model, output_dict, self.l2_weight)
                training_time = self._check_time()

                if epoch >= skip_eval:
                    valid_result_dict, test_result_dict = None, None
                    if self.no_filter:
                        valid_result,grp_scores = (
                            evaluate(model, validation_data, self.metrics)
                            if validation_data is not None
                            else [-1.0] * len(self.metrics)
                        )
                        test_result,grp_scores = (
                            evaluate(model, test_data, self.metrics)
                            if test_data is not None
                            else [-1.0] * len(self.metrics)
                        )
                    else:
                        valid_result, valid_result_dict,grp_scores = (
                            eval_multi_combination(
                                model, validation_data, self.metrics, fix_one
                            )
                            if validation_data is not None
                            else [-1.0] * len(self.metrics)
                        )
                        test_result, test_result_dict,grp_scores = (
                            eval_multi_combination(
                                model, test_data, self.metrics, fix_one
                            )
                            if test_data is not None
                            else [-1.0] * len(self.metrics)
                        )

                    testing_time = self._check_time()

                    self.valid_results.append(valid_result)
                    self.test_results.append(test_result)
                    self.disc_results.append(output_dict["d_score"])

                    if self.no_filter:
                        logging.info(f"Group-wise scores {grp_scores}")
                        logging.info(
                            "Epoch %5d [%.1f s]\n validation= %s test= %s [%.1f s] "
                            % (
                                epoch + 1,
                                training_time,
                                format_metric(valid_result),
                                format_metric(test_result),
                                testing_time,
                            )
                            + ",".join(self.metrics)
                        )
                    else:
                        logging.info(f"Group-wise scores {grp_scores}")
                        logging.info(
                            "Epoch %5d [%.1f s]\t Average: validation= %s test= %s [%.1f s] "
                            % (
                                epoch + 1,
                                training_time,
                                format_metric(valid_result),
                                format_metric(test_result),
                                testing_time,
                            )
                            + ",".join(self.metrics)
                        )
                        for key in valid_result_dict:
                            logging.info(
                                "validation= %s test= %s "
                                % (
                                    format_metric(valid_result_dict[key]),
                                    format_metric(test_result_dict[key]),
                                )
                                + ",".join(self.metrics)
                                + " ("
                                + key
                                + ") "
                            )

                    if max(self.valid_results) == self.valid_results[-1]:
                        model.save_model()
                        for idx in fair_disc_dict:
                            fair_disc_dict[idx].save_model()

                if epoch < skip_eval:
                    logging.info("Epoch %5d [%.1f s]" % (epoch + 1, training_time))

        except KeyboardInterrupt:
            logging.info("Early stop manually")
            save_here = input("Save here? (1/0) (default 0):")
            if str(save_here).lower().startswith("1"):
                model.save_model()
                for idx in fair_disc_dict:
                    fair_disc_dict[idx].save_model()

        # Find the best validation result across iterations
        best_valid_score = max(self.valid_results)
        best_epoch = self.valid_results.index(best_valid_score)
        
        # prepare disc result string
        disc_info = self.disc_results[best_epoch]
        disc_info_str = ["{}={:.4f}".format(key, disc_info[key]) for key in disc_info]
        disc_info_str = ",".join(disc_info_str)
        logging.info(
            "Best Iter(validation)= %5d\t valid= %s test= %s [%.1f s] "
            % (
                best_epoch + 1,
                format_metric(self.valid_results[best_epoch]),
                format_metric(self.test_results[best_epoch]),
                self.time[1] - self.time[0],
            )
            + ",".join(self.metrics)
            + " "
            + disc_info_str
            + " AUC"
        )
        best_test_score = max(self.test_results)
        best_epoch = self.test_results.index(best_test_score)
        disc_info = self.disc_results[best_epoch]
        disc_info_str = ["{}={:.4f}".format(key, disc_info[key]) for key in disc_info]
        disc_info_str = ",".join(disc_info_str)
        logging.info(
            "Best Iter(test)= %5d\t valid= %s test= %s [%.1f s] "
            % (
                best_epoch + 1,
                format_metric(self.valid_results[best_epoch]),
                format_metric(self.test_results[best_epoch]),
                self.time[1] - self.time[0],
            )
            + ",".join(self.metrics)
            + " "
            + disc_info_str
            + " AUC"
        )
        model.load_model()
        for idx in fair_disc_dict:
            fair_disc_dict[idx].load_model()


class TrainDisc:
    def __init__(
        self,
        no_filter,
        disc_epoch,
        num_worker=1,
        check_epoch=1,
        lr_attack=1e-4,
        l2_attack=1e-4,
    ):
        self.no_filter = no_filter
        self.num_worker = num_worker
        self.disc_epoch = disc_epoch
        self.check_epoch = check_epoch
        self.time = None
        self.lr_attack = lr_attack
        self.l2_attack = l2_attack

    def _check_time(self, start=False):
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def train_discriminator(self, model, dp_dict, fair_disc_dict):

        """
        Train discriminator to evaluate the quality of learned embeddings
        :param model: trained model
        :param dp_dict: Data processors for train valid and test
        :param fair_disc_dict: fairness discriminator dictionary
        :return:
        """
        train_data = DataLoader(
            dp_dict["train"],
            batch_size=dp_dict["train"].batch_size,
            num_workers=self.num_worker,
            shuffle=True,
            collate_fn=dp_dict["train"].collate_fn,
        )
        test_data = DataLoader(
            dp_dict["test"],
            batch_size=dp_dict["test"].batch_size,
            num_workers=self.num_worker,
            pin_memory=True,
            collate_fn=dp_dict["test"].collate_fn,
        )
        self._check_time(start=True)

        feature_results = defaultdict(list)
        best_results = dict()
        try:
            for epoch in range(self.disc_epoch):
                self._check_time()
                output_dict = self.fit_disc(
                    model,
                    train_data,
                    fair_disc_dict,
                    epoch=epoch,
                    lr_attack=self.lr_attack,
                    l2_attack=self.l2_attack,
                )

                if self.check_epoch > 0 and (
                    epoch == 1 or epoch % (self.disc_epoch // 4) == 0
                ):
                    check_disc(output_dict)
                training_time = self._check_time()

                test_result_dict = evaluation_disc(
                    model, fair_disc_dict, test_data, dp_dict["train"], self.no_filter
                )
                d_score_dict = test_result_dict["d_score"]
                if epoch % (self.disc_epoch // 4) == 0:
                    logging.info("Epoch %5d [%.1f s]" % (epoch + 1, training_time))
                for f_name in d_score_dict:
                    if epoch % (self.disc_epoch // 4) == 0:
                        logging.info(
                            "{} AUC= {:.4f}".format(f_name, d_score_dict[f_name])
                        )
                    feature_results[f_name].append(d_score_dict[f_name])
                    if d_score_dict[f_name] == max(feature_results[f_name]):
                        best_results[f_name] = d_score_dict[f_name]
                        idx = dp_dict["train"].f_name_2_idx[f_name]
                        fair_disc_dict[idx].save_model()

        except KeyboardInterrupt:
            logging.info("Early stop manually")
            save_here = input("Save here? (1/0) (default 0):")
            if str(save_here).lower().startswith("1"):
                for idx in fair_disc_dict:
                    fair_disc_dict[idx].save_model()

        for f_name in best_results:
            logging.info("{} best AUC: {:.4f}".format(f_name, best_results[f_name]))

        for idx in fair_disc_dict:
            fair_disc_dict[idx].load_model()

    def fit_disc(
        self, model, batches, fair_disc_dict, epoch=-1, lr_attack=None, l2_attack=None
    ):
        """
        Train the discriminator
        :param model: model instance
        :param batches: train data in batches
        :param fair_disc_dict: fairness discriminator dictionary
        :param epoch: epoch number
        :param lr_attack: attacker learning rate
        :param l2_attack: l2 regularization weight for attacker
        :return: return the output of the last round
        """
        gc.collect()
        torch.cuda.empty_cache()

        for idx in fair_disc_dict:
            discriminator = fair_disc_dict[idx]
            logging.info("Discriminator Optimizer: Adam")
            discriminator.optimizer = torch.optim.Adam(
                discriminator.parameters(),
                lr=self.lr_attack,
                weight_decay=self.l2_attack,
            )
            discriminator.train()

        output_dict = dict()
        loss_acc = defaultdict(list)

        for batch in tqdm(
            batches,
            leave=False,
            desc="Epoch %5d" % (epoch + 1),
            ncols=100,
            mininterval=1,
        ):
            if self.no_filter:
                mask = [0] * model.num_features
                mask = np.asarray(mask)
            else:
                mask = get_filter_mask(model.num_features)

            batch = move_batch_to_gpu(batch)

            labels = batch["features"]
            if not self.no_filter:
                masked_disc_label = get_masked_disc(fair_disc_dict, labels, mask)
            else:
                masked_disc_label = get_masked_disc(fair_disc_dict, labels, mask + 1)

            # calculate recommendation loss + fair discriminator penalty
            uids = batch["X"] - 1
            vectors = model.apply_filter(model.uid_embeddings(uids), mask)
            output_dict["check"] = []

            # update discriminator
            if len(masked_disc_label) != 0:
                for idx, (discriminator, label) in enumerate(masked_disc_label):
                    discriminator.optimizer.zero_grad()
                    disc_loss = discriminator(vectors.detach(), label)
                    disc_loss.backward()
                    discriminator.optimizer.step()
                    loss_acc[discriminator.name].append(disc_loss.detach().cpu())

        for key in loss_acc:
            loss_acc[key] = np.mean(loss_acc[key])

        output_dict["loss"] = loss_acc
        return output_dict
