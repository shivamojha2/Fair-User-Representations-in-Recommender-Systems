import logging
import os
import numpy as np
import shutil
import sys
from datetime import datetime


def set_logging(save_dir: str):
    """
    Set logging

    Args:
        save_dir (str): Path to directory
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    frmat = "%(asctime)s %(message)s"

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=frmat,
        datefmt="[%y-%m-%d %H:%M:%S]",
    )

    fh = logging.FileHandler(f"{save_dir}/logs.txt")
    fh.setFormatter(logging.Formatter(frmat))
    logging.getLogger().addHandler(fh)


def format_metric(metric):
    """
    Format metric
    """
    if type(metric) is not tuple and type(metric) is not list:
        metric = [metric]
    format_str = []
    if type(metric) is tuple or type(metric) is list:
        for m in metric:
            if (
                type(m) is float
                or type(m) is np.float
                or type(m) is np.float32
                or type(m) is np.float64
            ):
                format_str.append("%.4f" % m)
            elif (
                type(m) is int
                or type(m) is np.int
                or type(m) is np.int32
                or type(m) is np.int64
            ):
                format_str.append("%d" % m)
    return ",".join(format_str)


def save_file(src, dest, prefix):
    """
    Save file
    """
    if not os.path.exists(dest):
        os.makedirs(dest)
    dest = os.path.join(
        dest,
        "{}_{}_{}".format(
            prefix,
            datetime.now().strftime("%Y-%m-%d"),
            os.path.splitext(src)[-1],
        ),
    )
    shutil.copy(src, dest)
