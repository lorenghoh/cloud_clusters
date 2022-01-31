import numpy as np
import pandas as pd

import scipy.ndimage.measurements as measure
import scipy.ndimage.morphology as morph

from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed

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
        ds["QP"][:] / 1e3,
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

    def get_bstruct(st2d=False):
        b_struct = morph.generate_binary_structure(3, 2)

        if st2d:
            # Remove 3D connectivity
            b_struct[0] = 0
            b_struct[-1] = 0

        return b_struct

    def coord_to_zxy(df):
        """
        Unravels the raveled index (index -> z, y, x) and creates corresponding
        columns in the resulting Pandas DataFrame.

        """
        index = df.coord.values.coord_to_zxy

        # TODO: Now that this is vectorized, I can speed up the post-processing
        df["z"], df["y"], df["x"] = np.unravel_index(index, (192, 512, 1536))

        return df

    def write_clusters(t, ds, ds_e, src):
        # TODO: pre-process cloud field to exclude 2d cloud slices with no core
        # We can use the fact that the core is guaranteed to be in a cloudy region
        cor_b, cld_b, rmt_b = sample_conditional_field(ds, ds_e)

        b_struct = get_bstruct(st2d=True)
        cl_map, _ = measure.label((cld_b | rmt_b), structure=b_struct)
        cl_label = cl_map.ravel()

        # Contiguous core field
        b_struct = get_bstruct(st2d=False)
        c_map, _ = measure.label(cor_b, structure=b_struct)
        c_label = c_map.ravel()

        # Parse different cloud fields
        c_cor_map = (c_map > 0) & cor_b  # Full 3D core map

        cld_map = (cl_map > 0) & cld_b
        rmt_map = (cl_map > 0) & ~(cld_b | cor_b)

        def _get_df(c_label, c_map, c_type):
            c_flag = c_map.ravel()

            # Extrac indices
            c_i = np.arange(len(c_label))
            c_i = c_i[c_flag > 0]

            _df = pd.DataFrame.from_dict(
                {
                    "coord": c_i,
                    "cid": c_label[c_flag > 0],
                    "type": np.ones(len(c_i), dtype=int) * c_type,
                }
            )

            return _df

        # 3D core coordinates
        df = coord_to_zxy(_get_df(c_label, c_cor_map, 1))

        # 2d cloud coordinates
        df_cl = pd.DataFrame(columns=["coord", "cid", "type"])
        fields = [(0, cld_map), (2, rmt_map)]
        for i, c_fld in fields:
            df_cl = pd.concat([df_cl, _get_df(cl_label, c_fld, i)])
        df_cl = coord_to_zxy(df_cl)

        def _worker(g):
            # Pick the first row and find it in the cloud field
            # It is guaranteed to be in the cloud field DataFrame
            _g = g.iloc[0]

            _df = df_cl[(df_cl.z == _g.z) & (df_cl.y == _g.y) & (df_cl.x == _g.x)]
            _df = _df.assign(cid=_g.cid)

            return pd.concat([g, _df])

        grp = df.groupby(["cid", "z"], as_index=False)
        with Parallel(n_jobs=20) as Pr:
            result = Pr(delayed(_worker)(g) for _, g in grp)
            df_out = pd.concat(result, ignore_index=True)

        file_name = f"{src}/clusters/cloud_cluster_{t:04d}.pq"
        df_out.drop(["z", "y", "x"], axis=1).to_parquet(file_name)

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
