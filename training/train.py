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
from utils.utility_funcs import save_file, set_logging


def training(config: Config):
    """
    Run training

    Args:
        config (Config): Instance of config file
    """
    # logging
    exp_out_dir = os.path.join(config.save_dir, config.exp_name)
    logging_dir = os.path.join(exp_out_dir, "logs")
    set_logging(save_dir=logging_dir)
    save_file(src=config.json_path, dest=exp_out_dir, prefix=config.exp_name)

    # GPU settings
    logging.info("# cuda devices: %d", torch.cuda.device_count())

    # Reproducibility settings
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # create data processor
    data_processor_dict = {}
    for stage in ["train", "valid", "test"]:
        if config.data_processor in ["RecDataset"]:
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
        else:
            raise ValueError("Unknown DataProcessor")

    print("DATA LOADING DONE!")

    all_file = os.path.join(config.dataset_path, config.dataset_dir, "all.tsv")
    all_df = pd.read_csv(all_file, sep="\t")
    feature_cols = [
        name for name in all_df.columns.tolist() if name not in ["uid", "iid", "label"]
    ]
    rec_model_kwargs = {
        "user_num": len(set(all_df["uid"].tolist())),
        "item_num": len(set(all_df["iid"].tolist())),
        "u_vector_size": config.rec_model_kwargs["u_vector_size"],
        "i_vector_size": config.rec_model_kwargs["i_vector_size"],
        "feature_columns": feature_cols,
    }

    # Initialize Recommender system model
    rec_model = RecommenderModelZoo(
        config.rec_model_name, data_processor_dict, **rec_model_kwargs
    )

    # Initialize Discriminator model
    feature_cols = [
        name for name in all_df.columns.tolist() if name not in ["uid", "iid", "label"]
    ]
    Feature = namedtuple("Feature", ["num_class", "label_min", "label_max", "name"])
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

    fair_discriminator_dict = {}
    for feat_idx in feature_info:
        disc_model_kwargs = {
            "embed_dim": config.disc_model_kwargs["u_vector_size"],
            "feature_info": feature_info[feat_idx],
        }
        fair_discriminator_dict[feat_idx] = DiscriminatorModelZoo(
            config.disc_model_name, **disc_model_kwargs
        )

    print("Models Initialized")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", type=str, dest="json_path", help="Path to json config")
    args = parser.parse_args()

    config = Config(json_path=args.json_path)
    training(config)
