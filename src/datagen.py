"""
Datagen
"""
import logging
import os
import pickle
from collections import OrderedDict, defaultdict, namedtuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

Feature = namedtuple("Feature", ["num_class", "label_min", "label_max", "name"])


class BaseDatagen(Dataset):
    """
    Base Datagen for Recommender

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self, path, dataset_name, sep="\t", seq_sep=","):
        super().__init__()
        self.dataset_path = os.path.join(path, dataset_name)
        self.dataset_name = dataset_name
        self.sep = sep
        self.seq_sep = seq_sep
        self.train_file = os.path.join(self.dataset_path, "train.tsv")
        self.validation_file = os.path.join(self.dataset_path, "validation.tsv")
        self.test_file = os.path.join(self.dataset_path, "test.tsv")
        self.all_file = os.path.join(self.dataset_path, "all.tsv")
        self._load_data()

    def __len__(self):
        pass

    def __getitem__(self, index: int):
        pass

    def _load_data(self):
        if os.path.exists(self.all_file):
            logging.info("Load all.tsv")
            self.all_df = pd.read_csv(self.all_file, sep=self.sep)
        else:
            raise FileNotFoundError("All file is not found.")
        if os.path.exists(self.train_file):
            logging.info("Load train.tsv")
            self.train_df = pd.read_csv(self.train_file, sep=self.sep)
            logging.info("Size of train: %s", len(self.train_df))
        else:
            raise FileNotFoundError("train file is not found.")
        if os.path.exists(self.validation_file):
            logging.info("Load validation.tsv")
            self.validation_df = pd.read_csv(self.validation_file, sep=self.sep)
            logging.info("Size of validation: %s", len(self.validation_df))
        else:
            raise FileNotFoundError("Validation file is not found.")
        if os.path.exists(self.test_file):
            logging.info("Load test.tsv")
            self.test_df = pd.read_csv(self.test_file, sep=self.sep)
            logging.info("Size of test: %s", len(self.test_df))
        else:
            raise FileNotFoundError("Test file is not found.")


class MovieLens1MDataset(BaseDatagen):
    """
    RecDataset
    """

    def __init__(
        self,
        path,
        dataset_name,
        stage,
        batch_size=128,
        num_neg=1,
        sep="\t",
        seq_sep=",",
    ):
        super().__init__(path, dataset_name, sep, seq_sep)
        self.user_ids_set = set(self.all_df["uid"].tolist())
        self.item_ids_set = set(self.all_df["iid"].tolist())
        self.num_nodes = len(self.user_ids_set) + len(self.item_ids_set)
        self.train_item2users_dict = self._prepare_item2users_dict(self.train_df)

        self.all_user2items_dict = self._prepare_user2items_dict(self.all_df)
        self.train_user2items_dict = self._prepare_user2items_dict(self.train_df)
        self.valid_user2items_dict = self._prepare_user2items_dict(self.validation_df)
        self.test_user2items_dict = self._prepare_user2items_dict(self.test_df)

        # Add feature info for discriminator and filters
        uid_iid_label = ["uid", "iid", "label"]
        self.feature_columns = [
            name for name in self.train_df.columns.tolist() if name not in uid_iid_label
        ]
        self.feature_info = OrderedDict(
            {
                idx
                + 1: Feature(
                    self.all_df[col].nunique(),
                    self.all_df[col].min(),
                    self.all_df[col].max(),
                    col,
                )
                for idx, col in enumerate(self.feature_columns)
            }
        )
        self.num_features = len(self.feature_columns)

        self.num_user = len(self.user_ids_set)
        self.num_item = len(self.item_ids_set)
        self.batch_size = batch_size
        self.stage = stage
        self.num_neg = num_neg

        # Prepare test/validation dataset
        valid_pkl_path = os.path.join(self.dataset_path, "validation.pkl")
        test_pkl_path = os.path.join(self.dataset_path, "test.pkl")
        if self.stage == "valid":
            if os.path.exists(valid_pkl_path):
                with open(valid_pkl_path, "rb") as file:
                    logging.info("Load validation data from pickle file.")
                    self.data = pickle.load(file)
            else:
                self.data = self._get_data()
                with open(valid_pkl_path, "wb") as file:
                    pickle.dump(self.data, file)
        elif self.stage == "test":
            if os.path.exists(test_pkl_path):
                with open(test_pkl_path, "rb") as file:
                    logging.info("Load test data from pickle file.")
                    self.data = pickle.load(file)
            else:
                self.data = self._get_data()
                with open(test_pkl_path, "wb") as file:
                    pickle.dump(self.data, file)
        else:
            self.data = self._get_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def _prepare_user2items_dict(_df):
        df_groups = _df.groupby("uid")
        user_sample_dict = defaultdict(set)
        for uid, group in df_groups:
            user_sample_dict[uid] = set(group["iid"].tolist())
        return user_sample_dict

    @staticmethod
    def _prepare_item2users_dict(_df):
        df_groups = _df.groupby("iid")
        user_sample_dict = defaultdict(set)
        for uid, group in df_groups:
            user_sample_dict[uid] = set(group["uid"].tolist())
        return user_sample_dict

    def _get_data(self):
        if self.stage == "train":
            return self._get_train_data()
        else:
            return self._get_vt_data()

    def _get_train_data(self):
        df = self.train_df
        df["sample_id"] = df.index
        columns_order = ["uid", "iid", "sample_id", "label"] + [
            f_col for f_col in self.feature_columns
        ]
        data = df[columns_order].to_numpy()
        return data

    def _get_vt_data(self):
        if self.stage == "valid":
            df = self.validation_df
            logging.info("Prepare validation data")
        elif self.stage == "test":
            df = self.test_df
            logging.info("Prepare test data")
        else:
            raise ValueError("Wrong stage in dataset.")
        df["sample_id"] = df.index
        columns_order = ["uid", "iid", "sample_id", "label"] + [
            f_col for f_col in self.feature_columns
        ]
        data = df[columns_order].to_numpy()

        total_batches = int((len(df) + self.batch_size - 1) / self.batch_size)
        batches = []
        for n_batch in tqdm(
            range(total_batches),
            leave=False,
            ncols=100,
            mininterval=1,
            desc="Prepare Batches",
        ):
            batch_start = n_batch * self.batch_size
            batch_end = min(len(df), batch_start + self.batch_size)

            real_batch_size = batch_end - batch_start

            batch = data[batch_start : batch_start + real_batch_size, :]

            inputs = np.asarray(batch)[:, 0:3]
            labels = np.asarray(batch)[:, 3]
            features = np.asarray(batch)[:, 4:]
            inputs = np.concatenate((inputs, features), axis=1)

            neg_samples = self._neg_samples_from_all(inputs, self.num_neg)
            neg_labels = np.asarray([0] * neg_samples.shape[0])

            tmp_sample = np.concatenate((inputs, neg_samples), axis=0)
            samples = torch.from_numpy(tmp_sample[:, 0:3])
            labels = torch.from_numpy(np.concatenate((labels, neg_labels), axis=0))
            features = torch.from_numpy(tmp_sample[:, 3:])

            feed_dict = {"X": samples, "label": labels, "features": features}
            batches.append(feed_dict)
        return batches

    def collate_fn(self, batch):
        if self.stage == "train":
            feed_dict = self._collate_train(batch)
        else:
            feed_dict = self._collate_vt(batch)
        return feed_dict

    def _collate_train(self, batch):
        inputs = np.asarray(batch)[:, 0:3]
        labels = np.asarray(batch)[:, 3]
        features = np.asarray(batch)[:, 4:]
        neg_samples = self._neg_sampler(inputs)
        neg_samples = np.insert(neg_samples, 0, inputs[:, 0], axis=1)
        neg_samples = np.insert(neg_samples, 2, inputs[:, 2], axis=1)
        neg_labels = np.asarray([0] * neg_samples.shape[0])
        neg_features = np.copy(features)
        assert len(inputs) == len(neg_samples)
        samples = torch.from_numpy(np.concatenate((inputs, neg_samples), axis=0))
        labels = torch.from_numpy(np.concatenate((labels, neg_labels), axis=0))
        features = torch.from_numpy((np.concatenate((features, neg_features), axis=0)))
        feed_dict = {"X": samples, "label": labels, "features": features}
        return feed_dict

    @staticmethod
    def _collate_vt(data):
        return data

    def _neg_sampler(self, batch):
        neg_items = np.random.randint(1, self.num_item, size=(len(batch), self.num_neg))
        for i, (user, _, _) in enumerate(batch):
            user_clicked_set = self.all_user2items_dict[user]
            for j in range(self.num_neg):
                while neg_items[i][j] in user_clicked_set:
                    neg_items[i][j] = np.random.randint(1, self.num_item)
        return neg_items

    def _neg_samples_from_all(self, batch, num_neg=-1):
        neg_items = None
        for idx, data in enumerate(batch):
            user = data[0]
            sample_id = data[2]
            features = data[3:]
            neg_candidates = list(self.item_ids_set - self.all_user2items_dict[user])
            if num_neg != -1:
                if num_neg <= len(neg_candidates):
                    neg_candidates = np.random.choice(
                        neg_candidates, num_neg, replace=False
                    )
                else:
                    neg_candidates = np.random.choice(
                        neg_candidates, len(neg_candidates), replace=False
                    )
            user_arr = np.asarray([user] * len(neg_candidates))
            id_arr = np.asarray([sample_id] * len(neg_candidates))
            feature_arr = np.tile(features, (len(neg_candidates), 1))
            neg_candidates = np.expand_dims(np.asarray(neg_candidates), axis=1)
            neg_candidates = np.insert(neg_candidates, 0, user_arr, axis=1)
            neg_candidates = np.insert(neg_candidates, 2, id_arr, axis=1)
            neg_candidates = np.concatenate((neg_candidates, feature_arr), axis=1)

            if neg_items is None:
                neg_items = neg_candidates
            else:
                neg_items = np.concatenate((neg_items, neg_candidates), axis=0)

        return neg_items


class DiscriminatorDataset:
    @staticmethod
    def parse_dp_args(parser):
        """
        DataProcessor related argument parser
        :param parser: argument parser
        :return: updated argument parser
        """
        parser.add_argument(
            "--disc_batch_size",
            type=int,
            default=7000,
            help="discriminator train batch size",
        )
        return parser

    def __init__(
        self,
        path,
        dataset_name,
        stage,
        batch_size=1000,
        sep="\t",
        seq_sep=",",
        test_ratio=0.1,
        num_neg=1,
    ):
        self.sep = sep
        self.seq_sep = seq_sep
        self.dataset_path = os.path.join(path, dataset_name)
        self.all_file = os.path.join(self.dataset_path, "all.tsv")
        self.train_attacker_file = os.path.join(self.dataset_path, "attacker_train.tsv")
        self.test_attacker_file = os.path.join(self.dataset_path, "attacker_test.tsv")
        self.all_df = pd.read_csv(self.all_file, sep="\t")

        # Add feature info for discriminator and filters
        uid_iid_label = ["uid", "iid", "label"]
        self.feature_columns = [
            name for name in self.all_df.columns.tolist() if name not in uid_iid_label
        ]

        self.feature_info = OrderedDict(
            {
                idx
                + 1: Feature(
                    self.all_df[col].nunique(),
                    self.all_df[col].min(),
                    self.all_df[col].max(),
                    col,
                )
                for idx, col in enumerate(self.feature_columns)
            }
        )
        self.f_name_2_idx = {
            f_name: idx + 1 for idx, f_name in enumerate(self.feature_columns)
        }
        self.num_features = len(self.feature_columns)
        if os.path.exists(self.train_attacker_file) and os.path.exists(
            self.test_attacker_file
        ):
            self.train_df = pd.read_csv(self.train_attacker_file, sep="\t")
            self.test_df = pd.read_csv(self.test_attacker_file, sep="\t")
        else:
            self.train_df, self.test_df = self._init_feature_df(self.all_df, test_ratio)

        self.stage = stage
        self.batch_size = batch_size
        self.data = self._get_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _init_feature_df(self, all_df, test_ratio):
        logging.info("Initializing attacker train/test file")
        feature_df = pd.DataFrame()
        all_df = all_df.sort_values(by="uid")
        all_group = all_df.groupby("uid")

        uid_list = []
        feature_list_dict = {key: [] for key in self.feature_columns}
        for uid, group in all_group:
            uid_list.append(uid)
            for key in feature_list_dict:
                feature_list_dict[key].append(group[key].tolist()[0])
        feature_df["uid"] = uid_list
        for _f in self.feature_columns:
            feature_df[_f] = feature_list_dict[_f]

        test_size = int(len(feature_df) * test_ratio)
        sign = True
        counter = 0
        while sign:
            test_set = feature_df.sample(n=test_size).sort_index()
            for feat in self.feature_columns:
                num_class = self.feature_info[self.f_name_2_idx[feat]].num_class
                val_range = set([i for i in range(num_class)])
                test_range = set(test_set[feat].tolist())
                if len(val_range) != len(test_range):
                    sign = True
                    break
                else:
                    sign = False
            print(counter)
            counter += 1

        train_set = feature_df.drop(test_set.index)
        train_set.to_csv(self.train_attacker_file, sep="\t", index=False)
        test_set.to_csv(self.test_attacker_file, sep="\t", index=False)
        return train_set, test_set

    def _get_data(self):
        if self.stage == "train":
            return self._get_train_data()
        else:
            return self._get_test_data()

    def _get_train_data(self):
        data = self.train_df.to_numpy()
        return data

    def _get_test_data(self):
        data = self.test_df.to_numpy()
        return data

    @staticmethod
    def collate_fn(data):
        feed_dict = dict()
        feed_dict["X"] = torch.from_numpy(np.asarray(data)[:, 0])
        feed_dict["features"] = torch.from_numpy(np.asarray(data)[:, 1:])
        return feed_dict
