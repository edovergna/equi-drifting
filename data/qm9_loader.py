#!/usr/bin/python
# -*- coding:utf-8 -*-
import os

import numpy as np
import torch
import torch.nn.functional as F


class MMAPDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        mmap_dir: str,
        specify_data: Optional[str] = None,
        specify_index: Optional[str] = None,
        approx_length: int = 1,
        name: Optional[str] = None,
    ) -> None:
        super().__init__()

        self._indexes = []
        self._properties = []
        _index_path = (
            os.path.join(mmap_dir, "index.txt")
            if specify_index is None
            else specify_index
        )
        with open(_index_path, "r") as f:
            for line in f.readlines():
                messages = line.strip().split("\t")
                _id, start, end = messages[:3]
                _property = messages[3:]
                self._indexes.append((_id, int(start), int(end)))
                self._properties.append(_property)
        _data_path = (
            os.path.join(mmap_dir, "data.bin") if specify_data is None else specify_data
        )
        self._data_file = open(_data_path, "rb")
        self._mmap = mmap.mmap(self._data_file.fileno(), 0, access=mmap.ACCESS_READ)
        self.approx_length = approx_length
        self.name = name or ""

    def __del__(self):
        self._mmap.close()
        self._data_file.close()

    def __len__(self):
        return len(self._indexes)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        _, start, end = self._indexes[idx]
        data = decompress(self._mmap[start:end])
        if "label" not in data:
            data["label"] = 0

        return data

    @classmethod
    def collate_fn(cls, batch):
        keys = ["X", "B", "A", "atom_positions", "block_lengths", "segment_ids"]
        types = [
            torch.float,
            torch.long,
            torch.long,
            torch.long,
            torch.long,
            torch.long,
        ]
        res = {}
        for key, _type in zip(keys, types):
            val = []
            for item in batch:
                val.append(torch.tensor(item[key], dtype=_type))
            res[key] = torch.cat(val, dim=0)
        res["label"] = torch.tensor(
            [item["label"] for item in batch], dtype=torch.float
        )
        lengths = [len(item["B"]) for item in batch]
        res["lengths"] = torch.tensor(lengths, dtype=torch.long)
        res["X"] = res["X"].unsqueeze(-2)  # number of channel is 1
        # res['ids'] = [item['id'] for item in batch]
        return res


class QM9Dataset(MMAPDataset):

    target_properties = [
        "mu",
        "alpha",
        "homo",
        "lumo",
        "gap",
        "r2",
        "zpve",
        "U0",
        "U",
        "H",
        "G",
        "Cv",
    ]

    qm9_to_eV = {
        "U0": 27.2114,
        "U": 27.2114,
        "G": 27.2114,
        "H": 27.2114,
        "zpve": 27211.4,
        "gap": 27.2114,
        "homo": 27.2114,
        "lumo": 27.2114,
    }

    def __init__(self, mmap_dir: str, property: str) -> None:
        super().__init__(mmap_dir)

        prop_idx = self.target_properties.index(property)
        self._properties = [
            float(x[prop_idx]) for x in self._properties
        ]  # number of blocks in each data
        self.unit = 1 if property not in self.qm9_to_eV else self.qm9_to_eV[property]

    def get_item_len(self, idx: int):
        return self._properties[idx]

    def __getitem__(self, idx: int):
        """
        an example of the returned data
        {
            'X': [Natom, 3],
            'B': [Nblock],
            'A': [Natom],
            'atom_positions': [Natom],
            'block_lengths': [Nblock]
            'segment_ids': [Nblock],
        }
        """
        item = super().__getitem__(idx)
        item["label"] = [self._properties[idx] * self.unit]
        return item

    @classmethod
    def collate_fn(cls, batch):
        results = {
            "X": torch.cat(
                [torch.tensor(item["X"], dtype=torch.float) for item in batch], dim=0
            ),
            "B": torch.cat(
                [torch.tensor(item["B"], dtype=torch.long) for item in batch], dim=0
            ),
            "A": torch.cat(
                [torch.tensor(item["A"], dtype=torch.long) for item in batch], dim=0
            ),
            "atom_positions": torch.cat(
                [
                    torch.tensor(item["atom_positions"], dtype=torch.long)
                    for item in batch
                ],
                dim=0,
            ),
            "block_lengths": torch.cat(
                [
                    torch.tensor(item["block_lengths"], dtype=torch.long)
                    for item in batch
                ],
                dim=0,
            ),
            "segment_ids": torch.cat(
                [torch.tensor(item["segment_ids"], dtype=torch.long) for item in batch],
                dim=0,
            ),
            "lengths": torch.tensor(
                [len(item["B"]) for item in batch], dtype=torch.long
            ),
            "label": torch.cat(
                [torch.tensor(item["label"], dtype=torch.float) for item in batch],
                dim=0,
            ),
        }

        results["X"] = results["X"].unsqueeze(-2)  # number of channel is 1
        return results


if __name__ == "__main__":
    import sys

    dataset = QM9Dataset(sys.argv[1])
    print(dataset[0])
