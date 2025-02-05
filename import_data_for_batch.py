import os
import pathlib
import typing 
import functools
import pickle
import shutil

import numpy as np
from tqdm import tqdm
import polars as pl


class BatchData(typing.NamedTuple):
    f0: np.ndarray
    f1: np.ndarray
    f0_minus_f1: np.ndarray


MARKER_ID = ("Marker_Name", "Chromosome", "Position")
F0F1_NAME = ("f0", "f1")


def _ensure_path(path: typing.Union[pathlib.Path, str]) -> pathlib.Path:
    if isinstance(path, pathlib.Path):
        return path
    return pathlib.Path(path)


MADV_HUGEPAGE = 14


def _madvice_huge_page(array):
    return #a utiliser sous linux uniquement 
    import ctypes

    madvise = ctypes.CDLL("libc.so.6").madvise
    madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    madvise.restype = ctypes.c_int
    res = madvise(
        array.ctypes.data - array.offset,
        array.size * array.dtype.itemsize + array.offset,
        MADV_HUGEPAGE,
    )
    assert res == 0, f"MADVISE FAILED: res={res!r}"  # 1 means MADV_RANDOM


def _batches_from_dirpath(
    path: pathlib.Path, *, mode="r", shapes=None, dtype=np.float64
):
    if mode != "r":
        with open(path / "shapes.pkl", "wb") as f:
            pickle.dump(shapes, f)
        global_shape = (
            len(shapes),
            3,
            max(x for x, _ in shapes),
            max(x for _, x in shapes),
        )
    else:
        with open(path / "shapes.pkl", "rb") as f:
            shapes = pickle.load(f)
        global_shape = None
    array = np.lib.format.open_memmap(
        path / "data.npy", mode=mode, dtype=dtype, shape=global_shape
    )
    _madvice_huge_page(array)
    return [
        BatchData(
            f0=arr[0, : shape[0], : shape[1]],
            f1=arr[1, : shape[0], : shape[1]],
            f0_minus_f1=arr[2, : shape[0], : shape[1]],
        )
        for arr, shape in zip(array, shapes)
    ]


def build_loadable_data(
    input_path: pathlib.Path,
    prov_path: pathlib.Path,
    output_path: pathlib.Path,
    n_batches: int,
    dtype=np.float64,
    input_format: str = "csv",
):
    if input_format == "csv":
        read_smth, scan_smth = pl.read_csv, pl.scan_csv
    elif input_format == "parquet":
        read_smth, scan_smth = pl.read_parquet, pl.scan_parquet
    else:
        raise Exception

    input_path = _ensure_path(input_path)
    prov_path = _ensure_path(prov_path)
    output_path = _ensure_path(output_path)

    shutil.rmtree(prov_path, ignore_errors=True) #TODO: tester si le rep existe deja, abort
    os.mkdir(prov_path)
    shutil.rmtree(output_path, ignore_errors=True) #TODO: tester si le rep existe deja, abort 
    os.mkdir(output_path)

    files = [f for f in os.listdir(input_path) if f.endswith(input_format)]
    #files = os.listdir(input_path)#TODO : a filtrer par extension
    files.sort()
    pl.DataFrame({"file": files}).write_parquet(output_path / "columns.parquet")

    n_cols = len(files)
    common_markers = functools.reduce(
        lambda x, y: x.join(y, on=MARKER_ID, how="inner", coalesce=True),
        tqdm(
            (
                scan_smth(input_path / file).select(MARKER_ID).collect()
                for file in files
            ),
            total=len(files),
            desc="Common markers",
        ),
    )
    common_markers_sorted = (
        read_smth(input_path / files[0])
        .with_row_index()
        .join(common_markers, how="inner", coalesce=True, on=MARKER_ID)
        .sort("index")
        .select(MARKER_ID)
    )
    common_markers_sorted.write_parquet(output_path / "markers.parquet")

    for c, file in enumerate(tqdm(files, desc="Split")):
        local_path = prov_path / f"{c:08d}"
        os.mkdir(local_path)
        col_content = (
            common_markers_sorted.with_row_index()
            .join(
                read_smth(input_path / file), how="inner", coalesce=True, on=MARKER_ID
            )
            .sort("index")
            .select(("index",) + F0F1_NAME)
        )
        dfs = (
            col_content.with_columns(
                (col_content["index"] % n_batches).alias("batch_idx")
            )
            .select(("batch_idx",) + F0F1_NAME)
            .partition_by("batch_idx", include_key=False, as_dict=True)
        )

        for i in range(n_batches):
            dfs[i,].write_parquet(local_path / f"{i:08d}.parquet")

    batches_sizes = tuple(
        len(range(id_batch, len(common_markers_sorted), n_batches))
        for id_batch in range(n_batches)
    )
    shapes = [(x, n_cols) for x in batches_sizes]

    batches = _batches_from_dirpath(output_path, mode="w+", dtype=dtype, shapes=shapes)

    for id_batch, batch in enumerate(tqdm(batches, desc="Batches")):
        alldf = [
            pl.read_parquet(prov_path / f"{c:08d}" / f"{id_batch:08d}.parquet")
            for c, _ in enumerate(files)
        ]
        batch.f0[:, :] = np.array([df[F0F1_NAME[0]].to_numpy() for df in alldf]).T
        batch.f1[:, :] = np.array([df[F0F1_NAME[1]].to_numpy() for df in alldf]).T
        batch.f0_minus_f1[:, :] = batch.f0 - batch.f1


class LoadableData(typing.NamedTuple):
    markers: pl.DataFrame
    columns: typing.List[str]
    batches: typing.List[BatchData]


def load_loadable_data(path: pathlib.Path) -> LoadableData:
    path = _ensure_path(path)
    markers = pl.read_parquet(path / "markers.parquet")
    columns = list(pl.read_parquet(path / "columns.parquet")["file"])
    batches = _batches_from_dirpath(path)
    return LoadableData(markers, columns, batches)



