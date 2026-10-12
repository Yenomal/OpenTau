"""Small msgpack codec compatible with the OpenPI UniVTAC client."""

from __future__ import annotations

import functools

import msgpack
import numpy as np


def _pack_array(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("O", "c"):
            raise ValueError(f"Unsupported ndarray dtype: {value.dtype}")
        dtype = value.dtype.name if value.dtype.name == "bfloat16" else value.dtype.str
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": dtype,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    raise TypeError(f"Cannot msgpack value of type {type(value)!r}")


def _unpack_array(value):
    if b"__ndarray__" in value:
        dtype_name = value[b"dtype"]
        dtype = np.dtype("bfloat16") if dtype_name == "bfloat16" else np.dtype(dtype_name)
        return np.ndarray(buffer=value[b"data"], dtype=dtype, shape=tuple(value[b"shape"]))
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


Packer = functools.partial(msgpack.Packer, default=_pack_array)
packb = functools.partial(msgpack.packb, default=_pack_array, use_bin_type=True)


def unpackb(data: bytes):
    return msgpack.unpackb(data, object_hook=_unpack_array, raw=False)
