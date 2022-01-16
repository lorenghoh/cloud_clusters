from typing import ValuesView
import xarray as xr
import zarr as zr
import bsddb3 as bdb


def get_nc(item):
    with xr.open_dataset(item) as ds:
        try:
            ds = ds.squeeze("time")
        except Exception:
            pass

        return ds.load()


def get_zarr(item):
    with xr.open_zarr(item) as ds:
        try:
            ds = ds.squeeze("time")
        except Exception:
            pass

        return ds.load()


def get_db(item):
    with zr.DBMStore(item, open=bdb.btopen) as store:
        try:
            ds = xr.open_zarr(store)
            ds = ds.squeeze("time")
        except Exception:
            pass

        return ds.load()


def open_db(item):
    try:
        if item.suffix == ".nc":
            return get_nc(item)
        elif item.suffix == ".zarr":
            return get_zarr(item)
        elif item.suffix == ".db":
            return get_db(item)
        else:
            raise ValuesView("File type not recognized")
    except:
        raise
