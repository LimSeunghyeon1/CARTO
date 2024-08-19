import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

import multiprocessing
from concurrent import futures
import tqdm
import functools
import gc
import traceback
import logging

from typing import Dict, Any, Callable, List
import itertools

import trimesh
import urdfpy
import numpy as np

import uuid

import tyro

import dataclasses

from CARTO.lib import partnet_mobility
from CARTO.lib.partnet_mobility import PartNetMobilityV0DB, PartNetMobilityV0
from CARTO.simnet.lib.datapoint import compress_datapoint, decompress_datapoint

from CARTO.Decoder import utils, config
from CARTO.Decoder.data import dataset


def get_datapoint(
    object_id,
    urdf_object,
    cfg: config.GenerationConfig,
    joint_config={},
    transform=np.eye,
    debug_print: bool = False,
):
    object_trimesh, _, scale = utils.object_to_trimesh(
        urdf_object,
        joint_config=joint_config,
        base_transform=transform,
        origin_frame=cfg.origin_frame,
    )
    points, sdf = utils.object_to_sdf(
        object_trimesh, number_samples=cfg.sdf_number_samples
    )

    pc_points, pc_normals = utils.object_to_point_cloud(
        object_trimesh, number_samples=cfg.pc_number_samples
    )

    datapoint_id = uuid.uuid4()
    datapoint = dataset.DataPoint(
        object_id=object_id,
        joint_config_id=str(datapoint_id),
        scale=scale,
        sdf_values=sdf,
        points=points,
        joint_config=joint_config,
        full_pc=pc_points,
        full_normals=pc_normals,
    )
    gc.collect()
    return datapoint


def save_to_disk(path, datapoint: dataset.DataPoint):
    buf = compress_datapoint(datapoint)
    file_path = os.path.join(
        path, f"{datapoint.object_id}_{str(datapoint.joint_config_id)}.pickle.zstd"
    )
    with open(file_path, "wb") as fh:
        fh.write(buf)
    gc.collect()


def save_to_disk_future(path, future_):
    datapoint = future_.result()
    save_to_disk(path, datapoint)


def process_object_id(
    object_id: str,
    save_path: str,
    cfg: config.GenerationConfig,
    joint_filter: Callable[[Dict[str, Any]], bool] = lambda _: True,
    parallel_executor=None,
) -> List[futures.Future]:
    object_meta = PartNetMobilityV0DB.get_object_meta(object_id)
    object_path = PartNetMobilityV0DB.get_object(object_id)

    urdf_object = urdfpy.URDF.load(os.path.join(object_path, "mobility.urdf"))

    joints_of_interest: List[str] = []
    for joint_id, joint in object_meta["joints"].items():
        if not joint_filter(
            joint, partnet_mobility.get_joint_name_exclusion_list(object_meta)
        ):
            continue
        joints_of_interest.append(joint_id)

    if len(joints_of_interest) > 1:
        print(f"Skipping object with {len(joints_of_interest)} joints of interest")
        return []

    joint_configs_to_render = []
    for joint_steps_state in itertools.product(
        *[range(cfg.num_configs)] * len(joints_of_interest)
    ):
        joint_config = {}
        for joint_id, joint_step in zip(joints_of_interest, joint_steps_state):
            limits = partnet_mobility.get_canonical_joint_limits(object_meta, joint_id)
            joint_range = limits[1] - limits[0]
            joint_config[joint_id] = limits[0] + (
                (joint_range * joint_step / (cfg.num_configs - 1))
                if cfg.num_configs > 1
                else joint_range / 2
            )

        joint_configs_to_render.append(joint_config)

    canonical_transform = object_meta["canonical_transformation"]

    if parallel_executor is not None:
        all_futures = []
        for joint_config in joint_configs_to_render:
            future = parallel_executor.submit(
                get_datapoint,
                object_id,
                urdf_object,
                cfg,
                joint_config=joint_config,
                transform=canonical_transform,
            )
            save_to_disk_ = functools.partial(save_to_disk_future, save_path)
            future.add_done_callback(save_to_disk_)

            all_futures.append(future)
        return all_futures
    else:
        for joint_config in tqdm.tqdm(joint_configs_to_render):
            datapoint = get_datapoint(
                object_id,
                urdf_object,
                cfg,
                joint_config=joint_config,
                transform=canonical_transform,
            )
            save_to_disk(save_path, datapoint)
        return []


def main(args: config.GenerationConfig):
    object_filter, joint_filter = partnet_mobility.get_filter_function(
        category_list=args.categories,
        max_unique_parents=args.max_unique_parents,
        no_limit_ok=args.no_limit_ok,
        min_prismatic=args.min_prismatic,
        min_revolute=args.min_revolute,
        max_joints=args.max_joints,
        allowed_joints=args.allowed_joints,
    )
    partnet_mobility_db = PartNetMobilityV0()

    if args.id_file:
        object_ids: List[str] = open(args.id_file, "r").read().splitlines()
        object_filter = partnet_mobility.get_instance_filter(object_ids)

    print("object filter", object_filter)
    partnet_mobility_db.set_filter(object_filter)
    print(f"Length of filtered dataset: {len(partnet_mobility_db)}")

    suffix = f"_{args.suffix}" if args.suffix != "" else ""
    prefix = f"{args.prefix}_" if args.prefix != "" else ""

    save_path = os.path.join(config.BASE_DIR, "generated_data")
    if args.id_file:
        save_path = os.path.join(
            save_path,
            f"{prefix}{os.path.splitext(os.path.basename(args.id_file))[0]}_{args.num_configs}{suffix}",
        )
    else:
        save_path = os.path.join(
            save_path, f"{prefix}{'_'.join(args.categories)}_{args.num_configs}{suffix}"
        )
    os.makedirs(save_path, exist_ok=True)

    def save_cfg():
        with open(os.path.join(save_path, "config.yaml"), "w") as file:
            file.write(tyro.to_yaml(args))

    save_cfg()

    mp_context = multiprocessing.get_context("forkserver")

    for object_id in tqdm.tqdm(partnet_mobility_db.index_list):
        if args.parallel:
            divider = 1
            retry = True
            while retry:
                retry = False

                with futures.ProcessPoolExecutor(
                    max_workers=(args.max_workers // divider), mp_context=mp_context
                ) as parallel_executor:
                    try:
                        all_futures = process_object_id(
                            object_id,
                            save_path,
                            args,
                            joint_filter=joint_filter,
                            parallel_executor=parallel_executor,
                        )

                        with tqdm.tqdm(total=len(all_futures), leave=False) as pbar:
                            for future in futures.as_completed(all_futures):
                                pbar.update(1)
                                datapoint: dataset.DataPoint = future.result()
                                args.max_extent = float(
                                    max(
                                        np.max(np.abs(datapoint.full_pc)),
                                        args.max_extent,
                                    )
                                )
                                save_cfg()

                        parallel_executor.shutdown(wait=True)
                        gc.collect()
                    except Exception as e:
                        logging.error(traceback.format_exc())
                        print(f"Error processing {object_id}")
                        divider *= 2
                        if divider > args.max_workers:
                            retry = False
                            continue
                        print(
                            f"As memory overflowed, Re-trying with less workers --> dividing (//) workers by {divider = }"
                        )
                        retry = True
                        already_saved = [
                            f
                            for f in os.listdir(save_path)
                            if f.startswith(f"{object_id}_")
                        ]
                        for file_ in already_saved:
                            os.remove(os.path.join(save_path, file_))
        else:
            process_object_id(
                object_id,
                save_path,
                args,
                joint_filter=joint_filter,
                parallel_executor=None,
            )


if __name__ == "__main__":
    args: config.GenerationConfig = tyro.parse(config.GenerationConfig)
    main(args)
