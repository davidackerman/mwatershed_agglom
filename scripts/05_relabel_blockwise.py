import json
import hashlib
import logging
import numpy as np
import os
import daisy
import pymongo
import time
import subprocess

from funlib.persistence import open_ds, prepare_ds
from funlib.geometry import Coordinate
from pathlib import Path

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

def relabel_dataset(
    sample_name: str,
    fragments_file,
    fragments_dataset,
    block_size,
    db_host,
    db_name,
    num_workers,
    mask_file,
    mask_dataset,
    drop=False,
    billing=None,
):

    # prepare fragments dataset
    logging.info("Reading fragments from %s", fragments_file)
    fragments = open_ds(fragments_file, fragments_dataset, mode="r")
    lut = f'luts_full/seg_{sample.replace("/","-")}_edges_mwatershed.npz'
    mapping = np.load(Path(fragments_file, lut))["fragment_segment_lut"]

    # max_value = np.max(mapping[1, ])
    # if max_value <= np.iinfo(np.uint8).max:
    #     dtype = np.uint8
    # elif max_value <= np.iinfo(np.uint16).max:
    #     dtype = np.uint16
    # elif max_value <= np.iinfo(np.uint32).max:
    #     dtype = np.uint32
    # else:
    #     dtype = np.uint64

    # For now have to do uint64 since the nodes aren't saved so don't know their values and can't replace them
    dtype = np.uint64

    fragments_relabeled_dataset = f"{fragments_dataset}_relabeled"
    prepare_ds(
        fragments_file,
        fragments_relabeled_dataset,
        fragments.roi,
        fragments.voxel_size,
        dtype,
        daisy.Roi((0, 0, 0), block_size),
        delete=drop,
    )

    total_roi = fragments.roi
    read_roi = write_roi = daisy.Roi((0,) * fragments.roi.dims, block_size)


    client = pymongo.MongoClient(db_host)
    db = client[db_name]

    completed_collection_name = f"{sample_name}_fragment_blocks_relabeled"
    completed_collection = None

    if completed_collection_name in db.list_collection_names():
        completed_collection = db[completed_collection_name]
        if drop:
            print(f"dropping {completed_collection}")
            db.drop_collection(completed_collection)
    
    if completed_collection_name not in db.list_collection_names():
        completed_collection = db[completed_collection_name]
        completed_collection.create_index(
            [("block_id", pymongo.ASCENDING)], name="block_id"
        )

    complete_cache = set(
        [tuple(doc["block_id"]) for doc in completed_collection.find()]
    )

    extract_fragments_task = daisy.Task(
        f"{sample_name}_relabel_fragments",
        total_roi=total_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=lambda: start_worker(
            sample_name,
            fragments_file,
            fragments_dataset,
            db_host,
            db_name,
            mask_file,
            mask_dataset,
            billing,
        ),
        check_function=None,#lambda b: check_block(completed_collection, complete_cache, b),
        num_workers=num_workers,
        read_write_conflict=False,
        fit="shrink",
    )

    daisy.run_blockwise([extract_fragments_task])


def start_worker(
    sample_name: str,
    fragments_file,
    fragments_dataset,
    db_host,
    db_name,
    mask_file,
    mask_dataset,
    billing,
):
    worker_id = daisy.Context.from_env()["worker_id"]
    task_id = daisy.Context.from_env()["task_id"]

    logging.info("worker %s started...", worker_id)

    output_basename = daisy.get_worker_log_basename(worker_id, task_id)

    log_out = output_basename.parent / f"worker_{worker_id}.out-bsub"
    log_err = output_basename.parent / f"worker_{worker_id}.err-bsub"

    config = {
        "sample_name": sample_name,
        "fragments_file": fragments_file,
        "fragments_dataset": fragments_dataset,
        "db_host": db_host,
        "db_name": db_name,
        "mask_file": mask_file,
        "mask_dataset": mask_dataset,
    }
    config_str = "".join(["%s" % (v,) for v in config.values()])
    config_hash = abs(int(hashlib.md5(config_str.encode()).hexdigest(), 16))

    config_file = os.path.join(output_basename.parent, "%d.config" % config_hash)

    with open(config_file, "w") as f:
        json.dump(config, f)

    logging.info("Running block with config %s..." % config_file)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    worker = f"{dir_path}/05_relabel_worker.py"

    command = f"python {worker} {config_file}"
    subprocess.run(
        ["bsub", "-P", billing, "-n", "4", "-o", log_out, "-e", log_err, command]
    )


def check_block(completed_collection, complete_cache, block):
    done = (
        block.block_id in complete_cache
        or len(list(completed_collection.find({"block_id": block.block_id}))) >= 1
    )

    return done


if __name__ == "__main__":
    voxel_size = Coordinate(8, 8, 8)
    block_size = Coordinate(128, 128, 128) * voxel_size
    context = Coordinate(16, 16, 16) * voxel_size
    start = time.time()

    sample = "2023-08-17/plasmodesmata_affs_lsds/0"
    relabel_dataset(
        sample_name=sample,
        fragments_file="/nrs/cellmap/ackermand/predictions/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.n5",
        fragments_dataset=f"processed/{sample}/fragments",
        block_size=tuple(block_size),
        db_host="mongodb://microdosingAdmin:Cu2CO3OH2@funke-mongodb2.int.janelia.org:27017",
        db_name="cellmap_postprocessing_ackermand",
        num_workers=300,
        mask_file=None,#"/nrs/cellmap/ackermand/cellmap/leaf-gall/prediction_masks.zarr",
        mask_dataset=None,#"jrc_22ak351-leaf-3m",
        drop=True,
        billing="cellmap",
    )

    end = time.time()

    seconds = end - start
    minutes = seconds / 60
    hours = minutes / 60
    days = hours / 24

    print(
        "Total time to extract fragments: %f seconds / %f minutes / %f hours / %f days"
        % (seconds, minutes, hours, days)
    )
