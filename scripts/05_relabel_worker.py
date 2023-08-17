from funlib.segment.arrays import relabel, replace_values
from funlib.geometry import Coordinate
from funlib.persistence import open_ds, Array, graphs

import mwatershed as mws
import daisy

import pymongo
from scipy.ndimage import measurements
import numpy as np
from scipy.ndimage.filters import gaussian_filter

import logging
import json
import sys
import time
from pathlib import Path

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def relabel_in_block(
    fragments,
    fragments_relabeled,
    mapping,
    block,
):
    """
    Args:
        filter_fragments (float):
            Filter fragments that have an average affinity lower than this
            value.
    """

    logger.info("reading affs from %s", block.read_roi)

    fragments = fragments.intersect(block.read_roi)
    fragments.materialize()
    fragments = fragments.data
    # store relabeled fragments
    fragments_relabeled[block.write_roi] = replace_values(fragments, mapping[0], mapping[1])#.astype(np.uint8)#fragments_relabeled.dtype)


def relabel_fragments_worker(input_config):
    
    logger.info(sys.argv)

    with open(input_config, "r") as f:
        config = json.load(f)

    logger.info(config)

    sample_name = config["sample_name"]
    fragments_file = config["fragments_file"]
    fragments_dataset = config["fragments_dataset"]
    db_name = config["db_name"]
    db_host = config["db_host"]
    if config["mask_file"] is not None:
        logger.info("Reading mask from %s", config["mask_file"])
        mask = open_ds(config["mask_file"], config["mask_dataset"], mode="r")
    else:
        mask = None

    logger.info("Reading affs from %s", fragments_file)
    fragments = open_ds(fragments_file, fragments_dataset, mode="r")

    lut = f'luts_full/seg_{sample_name.replace("/","-")}_edges_mwatershed.npz'
    logger.info("Reading lut from %s", Path(fragments_file) / Path(lut))
    mapping = np.load(Path(fragments_file, lut))["fragment_segment_lut"]

    fragments_relabeled_dataset = f"{fragments_dataset}_relabeled"
    logger.info("writing relabeled fragments to %s", fragments_relabeled_dataset)
    fragments_relabeled = open_ds(fragments_file, fragments_relabeled_dataset, mode="r+")

    # open block done DB
    mongo_client = pymongo.MongoClient(db_host)
    db = mongo_client[db_name]
    blocks_relabeled = db[f"{sample_name}_fragment_blocks_relabeled"]

    client = daisy.Client()

    while True:
        logger.info("getting block")
        with client.acquire_block() as block:
            logger.info(f"got block {block}")

            if block is None:
                break
            if mask:
                should_process_block = np.any(mask.to_ndarray(roi=block.write_roi.snap_to_grid(mask.voxel_size),fill_value=0))
                if not should_process_block:
                    continue
            #start = time.time()

            logger.info("block read roi begin: %s", block.read_roi.get_begin())
            logger.info("block read roi shape: %s", block.read_roi.get_shape())
            logger.info("block write roi begin: %s", block.write_roi.get_begin())
            logger.info("block write roi shape: %s", block.write_roi.get_shape())

            relabel_in_block(
                fragments,
                fragments_relabeled,
                mapping,
                block,
            )

            # document = {
            #     "block_id": block.block_id,
            #     "read_roi": (block.read_roi.get_begin(), block.read_roi.get_shape()),
            #     "write_roi": (block.write_roi.get_begin(), block.write_roi.get_shape()),
            #     "start": start,
            #     "duration": time.time() - start,
            # }
            # blocks_relabeled.insert_one(document)
            logger.info(f"releasing block: {block}")


if __name__ == "__main__":
    relabel_fragments_worker(sys.argv[1])
