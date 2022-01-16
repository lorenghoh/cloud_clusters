import xarray as xr
import zarr as zr
import bsddb3 as bdb


def get_nc(item):
    ds = xr.open_dataset(item)
    yield ds

    ds.close()


def get_zarr(item):
    ds = xr.open_zarr(item)
    yield ds

    ds.close()


def get_db(item):
    store = zr.DBMStore(item, open=bdb.btopen)
    yield xr.open_zarr(store)

    store.close()


def open_db(item):
    if item.suffix == ".nc":
        yield get_nc(item)
    elif item.suffix == ".zarr":
        yield get_zarr(item)
    elif item.suffix == ".db":
        yield get_db(item)
    else:
        raise ValueError("File type not recognized")
