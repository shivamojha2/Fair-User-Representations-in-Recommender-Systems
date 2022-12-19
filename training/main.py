"""
Training module
"""
import argparse
import logging
import os
import random
from collections import OrderedDict, namedtuple

import numpy as np
import pandas as pd
import torch
from config import Config
from datagen_zoo import DataGenZoo
from discriminator_model_zoo import DiscriminatorModelZoo
from recommender_model_zoo import RecommenderModelZoo
from utils.utility_funcs import save_file, set_logging, format_metric
from train import TrainModel, TrainDisc
from torch.utils.data import DataLoader
from evals import *

Feature = namedtuple("Feature", ["num_class", "label_min", "label_max", "name"])


def fair_user_rec_sys(config: Config):
    """
    Initialize dataloader andrun training

    Args:
        config (Config): Instance of config file
    """
    # logging
    exp_out_dir = os.path.join(config.save_dir, config.exp_name)
    logging_dir = os.path.join(exp_out_dir, "logs")
    set_logging(save_dir=logging_dir)
    save_file(src=config.json_path, dest=exp_out_dir, prefix=config.exp_name)

    # GPU settings
    logging.info("Number of cuda devices: %d", torch.cuda.device_count())

    # Reproducibility settings
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load data
    data_processor_dict = {}
    for stage in ["train", "valid", "test"]:
        if stage == "train":
            data_processor_dict[stage] = DataGenZoo(
                config.dataset,
                config.dataset_path,
                config.dataset_dir,
                stage,
                batch_size=config.train_datagen_kwargs["batch_size"],
                num_neg=config.train_datagen_kwargs["num_neg"],
            )
        else:
            data_processor_dict[stage] = DataGenZoo(
                config.dataset,
                config.dataset_path,
                config.dataset_dir,
                stage,
                batch_size=config.vt_datagen_kwargs["batch_size"],
                num_neg=config.vt_datagen_kwargs["num_neg"],
            )

    logging.info("Data loading complete.")

    # Processing input df
    all_file = os.path.join(config.dataset_path, config.dataset_dir, "all.tsv")
    all_df = pd.read_csv(all_file, sep="\t")
    feature_cols = [
        name for name in all_df.columns.tolist() if name not in ["uid", "iid", "label"]
    ]
    feature_cols = [
        name for name in all_df.columns.tolist() if name not in ["uid", "iid", "label"]
    ]
    feature_info = OrderedDict(
        {
            idx
            + 1: Feature(
                all_df[col].nunique(),
                all_df[col].min(),
                all_df[col].max(),
                col,
            )
            for idx, col in enumerate(feature_cols)
        }
    )

    # Initialize Recommender system model
    rec_model_kwargs = {
        "user_num": len(set(all_df["uid"].tolist())),
        "item_num": len(set(all_df["iid"].tolist())),
        "u_vector_size": config.rec_model_kwargs["u_vector_size"],
        "i_vector_size": config.rec_model_kwargs["i_vector_size"],
        "feature_columns": feature_cols,
    }
    rec_model = RecommenderModelZoo(
        config.rec_model_name, data_processor_dict, exp_out_dir, **rec_model_kwargs
    )

    # Initialize Discriminator model
    fair_discriminator_dict = {}
    for feat_idx in feature_info:
        disc_model_kwargs = {
            "embed_dim": config.disc_model_kwargs["u_vector_size"],
            "feature_info": feature_info[feat_idx],
        }
        fair_discriminator_dict[feat_idx] = DiscriminatorModelZoo(
            config.disc_model_name, exp_out_dir, **disc_model_kwargs
        )

    logging.info("Recommender system model and discriminators initialized.")

    if config.load_model_flag:
        rec_model.load_model()
        for idx in fair_discriminator_dict:
            fair_discriminator_dict[idx].load_model()

    model_trainer = TrainModel(
        feature_info=feature_info, **config.train_rec_model_kwargs
    )
    if config.train_rec_model:
        model_trainer.train(
            rec_model,
            data_processor_dict,
            fair_discriminator_dict,
            skip_eval=config.skip_eval,
        )

    # Reproducibility settings
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if config.eval_discriminator:
        # create extra data processor
        ex_data_processor_dict = {}
        for stage in ["train", "test"]:
            ex_data_processor_dict[stage] = DataGenZoo(
                config.disc_dataset,
                config.dataset_path,
                config.dataset_dir,
                stage,
                batch_size=config.train_datagen_kwargs["batch_size"],
                num_neg=config.train_datagen_kwargs["num_neg"],
            )

    # Creating discriminators
    ex_fair_discriminator_dict = {}
    for feat_idx in feature_info:
        upd_disc_model_kwargs = {
            "embed_dim": config.upd_disc_model_kwargs["u_vector_size"],
            "feature_info": feature_info[feat_idx],
            "model_name": "eval",
        }
        if feature_info[feat_idx].num_class == 2:
            ex_fair_discriminator_dict[feat_idx] = DiscriminatorModelZoo(
                config.binary_disc_model_name, exp_out_dir, **upd_disc_model_kwargs
            )
        else:
            ex_fair_discriminator_dict[feat_idx] = DiscriminatorModelZoo(
                config.multi_disc_model_name, exp_out_dir, **upd_disc_model_kwargs
            )

    if config.load_attacker_model_flag:
        for idx in ex_fair_discriminator_dict:
            logging.info("load attacker model...")
            ex_fair_discriminator_dict[idx].load_model()

    rec_model.load_model()
    rec_model.freeze_model()
    disc_trainer = TrainDisc(**config.attack_disc_model_kwargs)
    disc_trainer.train_discriminator(
        rec_model, ex_data_processor_dict, ex_fair_discriminator_dict
    )

    test_data = DataLoader(
        data_processor_dict["test"],
        batch_size=None,
        num_workers=1,
        pin_memory=True,
        collate_fn=data_processor_dict["test"].collate_fn,
    )

    test_result_dict = dict()
    if config.no_filter:
        test_result = evaluate(
            rec_model, test_data, (config.metrics).lower().split(",")
        )
        logging.info(
            "Test After Training = %s ",
            (format_metric(test_result))
            + ",".join((config.metrics).lower().split(",")),
        )
    else:
        test_result, test_result_dict = eval_multi_combination(
            rec_model,
            test_data,
            config.metrics,
        )
        logging.info(
            "Test Data Performance After Training:\t Average: %s ",
            (format_metric(test_result))
            + ",".join((config.metrics).lower().split(",")),
        )
        for key in test_result_dict:
            logging.info(
                "test= %s ",
                (format_metric(test_result_dict[key]))
                + ",".join((config.metrics).lower().split(","))
                + " ("
                + key
                + ") ",
            )
    logging.info("Script execution complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", type=str, dest="json_path", help="Path to json config")
    args = parser.parse_args()

    config = Config(json_path=args.json_path)
    fair_user_rec_sys(config)
