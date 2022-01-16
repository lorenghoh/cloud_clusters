import numpy as np
import pandas as pd

import xarray as xr
import zarr as zr
import bsddb3 as bdb

import scipy.ndimage.measurements as measure
import scipy.ndimage.morphology as morph

from tqdm import tqdm
from pathlib import Path

import lib.calcs
import lib.io
import lib.config

config = lib.config.read_config()


# Initialize config with project working directory
pwd = Path(__file__).absolute().parents[0]


def sample_conditional_field(ds, ds_e):
    """
    Define conditional fields

    Return
    ------
    cor_b: core field (QN > 0, W > 0, B > 0)
    cld_b: cloud field (QN > 0)
    rmt_b: ramnant of core entrainment
    TODO: trc_fld: plume (tracer-dependant)
    """
    th_v = lib.calcs.theta_v(
        ds["p"][:] * 100,
        ds["TABS"][:],
        ds["QV"][:] / 1e3,
        ds["QN"][:] / 1e3,
        ds["QP"][:] / 1e3
    )

    qn = (ds["QN"] > 0).values
    w = ((ds["W"] + np.roll(ds["W"], 1, axis=1)) / 2 > 0).values
    buoy = (th_v > np.nanmean(th_v, axis=(1, 2))[:, None, None]).values
    ent = (ds_e["ETETCOR"] > 0).values

    # Boolean maps
    cld_b = qn
    cor_b = w & buoy & cld_b
    rmt_b = ent

    # Define plume based on tracer fields (computationally intensive)
    # tr_field = ds['TR01'][:]
    # tr_ave = np.nanmean(tr_field, axis=(1, 2))
    # tr_std = np.std(tr_field, axis=(1, 2))
    # tr_min = .05 * np.cumsum(tr_std) / (np.arange(len(tr_std))+1)

    # c2_fld = (tr_field > \
    #             np.max(np.array([tr_ave + tr_std, tr_min]), 0)[:, None, None])

    return cor_b, cld_b, rmt_b


def cluster_clouds():
    """
    Clustering the cloud volumes/projections from the Zarr dataset.

    There are a few different ways to cluster cloud/core volumes. The possible
    choices are: 2d cluster (no clustering), ~~3d_volume (full cloud volume)~~,
    3d_projection (projections of 3D volumes), 2d_projection (projections of the
    entire cloud field), etc.

    Clustering options:
    - 2d: 2D cluster (horizontal slices)
    - projection: 3D projection

    Return
    ------
    Parquet files containing the coordinates
    Type:
        - 0 for cloudy cells
        - 1 for core cells
        - 2 for entrainment remnants
    """
    src = Path(config["src"])

    # Ensure that the pq folder exists
    if (src / "pq").exists == False:
        raise FileNotFoundError("pq folder not found. Ensure config.json validation")

    def write_clusters(t, ds, ds_e, src):
        bin_st = morph.generate_binary_structure(3, 2)

        # Remove 3D connectivity
        bin_st[0] = 0
        bin_st[-1] = 0

        cor_b, cld_b, rmt_b = sample_conditional_field(ds, ds_e)

        df = pd.DataFrame(columns=["coord", "cid", "type"])
        c_map, _ = measure.label((cld_b | rmt_b), structure=bin_st)
        c_label = c_map.ravel()

        # Parse different cloud fields
        cld_map = (c_map > 0) & cld_b
        cor_map = (c_map > 0) & cor_b
        rmt_map = (c_map > 0) & ~(cld_b | cor_b)

        target_fields = [cld_map, cor_map, rmt_map]
        for i, c_fld in enumerate(target_fields):
            c_flag = c_fld.ravel()

            # Extract indices
            c_index = np.arange(len(c_label))
            c_index = c_index[c_flag > 0]
            c_id = c_label[c_flag > 0]
            c_type = np.ones(len(c_id), dtype=int) * i

            df_ = pd.DataFrame.from_dict(
                {"coord": c_index, "cid": c_id, "type": c_type}
            )

            df = pd.concat([df, df_])

        file_name = f"{src}/pq/cloud_cluster_{t:04d}.pq"
        df.to_parquet(file_name)

        tqdm.write(f"Written {file_name}")

    ds_l = sorted((src / "variables").glob("CGILS_1728*"))
    de_l = sorted((src / "ent_core").glob("CGILS_CORE*"))

    if len(ds_l) != len(de_l):
        raise ValueError("Database integrity check failed")

    for i in tqdm(range(len(ds_l))):
        ds = next(next(lib.io.open_db(ds_l[i])))
        de = next(next(lib.io.open_db(de_l[i])))

        write_clusters(i, ds, de, src)

        ds.close()
        de.close()


if __name__ == "__main__":
    cluster_clouds()
