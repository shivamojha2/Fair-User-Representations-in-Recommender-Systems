import logging
import os
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
        os.mkdir(save_dir)

    frmat = "%(asctime)s %(message)s"

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=frmat,
        datefmt="[%y%m%d %H:%M:%S]",
    )

    fh = logging.FileHandler(f"{save_dir}/logs.txt")
    fh.setFormatter(logging.Formatter(frmat))
    logging.getLogger().addHandler(fh)


def save_file(src, dest, prefix):
    if not os.path.exists(dest):
        os.mkdir(dest)
    dest = os.path.join(
        dest,
        "{}_{}_{}".format(
            prefix,
            datetime.now().strftime("%Y-%m-%d"),
            os.path.splitext(src)[-1],
        ),
    )
    shutil.copy(src, dest)
