#!/usr/bin/python
# -*- coding:utf-8 -*-
from copy import copy
from typing import List


class MoleculeVocab:

    def __init__(self):

        self.backbone_atoms = ["N", "CA", "C", "O"]
        self.PAD, self.MASK, self.UNK, self.GLB = (
            "#",
            "*",
            "?",
            "&",
        )  # pad / mask / unk / global node
        specials = [  # special added
            (self.PAD, "PAD"),
            (self.MASK, "MASK"),
            (self.UNK, "UNK"),  # pad / mask / unk
            (self.GLB, "<G>"),  # global node
        ]
        aas = [  # amino acids (1-letter symbol, 3-letter abbreviation)
            ("G", "GLY"),
            ("A", "ALA"),
            ("V", "VAL"),
            ("L", "LEU"),
            ("I", "ILE"),
            ("F", "PHE"),
            ("W", "TRP"),
            ("Y", "TYR"),
            ("D", "ASP"),
            ("H", "HIS"),
            ("N", "ASN"),
            ("E", "GLU"),
            ("K", "LYS"),
            ("Q", "GLN"),
            ("M", "MET"),
            ("R", "ARG"),
            ("S", "SER"),
            ("T", "THR"),
            ("C", "CYS"),
            ("P", "PRO"),  # 20 aa
            # ('U', 'SEC') # 21 aa for eukaryote
        ]

        chemical_symbols = [  # Periodic Table
            # 1
            "H",
            "He",
            # 2
            "Li",
            "Be",
            "B",
            "C",
            "N",
            "O",
            "F",
            "Ne",
            # 3
            "Na",
            "Mg",
            "Al",
            "Si",
            "P",
            "S",
            "Cl",
            "Ar",
            # 4
            "K",
            "Ca",
            "Sc",
            "Ti",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Zn",
            "Ga",
            "Ge",
            "As",
            "Se",
            "Br",
            "Kr",
            # 5
            "Rb",
            "Sr",
            "Y",
            "Zr",
            "Nb",
            "Mo",
            "Tc",
            "Ru",
            "Rh",
            "Pd",
            "Ag",
            "Cd",
            "In",
            "Sn",
            "Sb",
            "Te",
            "I",
            "Xe",
            # 6
            "Cs",
            "Ba",
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Pm",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
            "Hf",
            "Ta",
            "W",
            "Re",
            "Os",
            "Ir",
            "Pt",
            "Au",
            "Hg",
            "Tl",
            "Pb",
            "Bi",
            "Po",
            "At",
            "Rn",
            # 7
            "Fr",
            "Ra",
            "Ac",
            "Th",
            "Pa",
            "U",
            "Np",
            "Pu",
            "Am",
            "Cm",
            "Bk",
            "Cf",
            "Es",
            "Fm",
            "Md",
            "No",
            "Lr",
            "Rf",
            "Db",
            "Sg",
            "Bh",
            "Hs",
            "Mt",
            "Ds",
            "Rg",
            "Cn",
            "Nh",
            "Fl",
            "Mc",
            "Lv",
            "Ts",
            "Og",
        ]

        # previous
        # sms = [  # small molecules (symbol, abbreviation - upper case)
        #         ('c', 'C'), ('n', 'N'), ('o', 'O'), ('s', 'S'),
        #         ('p', 'P'), ('b', 'B'), ('cl', 'CL'), ('f', 'F'),
        #         ('br', 'BR'), ('i', 'I')
        #     ]

        sms = [(e.lower(), e.upper()) for e in chemical_symbols]

        bases = [  # bases for RNA/DNA
            ("DA", "DA"),
            ("DG", "DG"),
            ("DC", "DC"),
            ("DT", "DT"),  # DNA
            ("R-A", "R-A"),
            ("R-G", "R-G"),
            ("R-C", "R-C"),
            ("R-U", "R-U"),  # RNA
        ]

        self.atom_pad, self.atom_mask, self.atom_global = (
            "pad",
            "msk",
            "glb",
        )  # Avoid conflict with atom P
        self.atom_pos_pad, self.atom_pos_mask, self.atom_pos_global = (
            "pad",
            "msk",
            "glb",
        )
        self.atom_pos_sm = "sml"  # small molecule

        # block level vocab
        self.idx2block = specials + aas + sms + bases
        self.symbol2idx, self.abrv2idx = {}, {}
        for i, (symbol, abrv) in enumerate(self.idx2block):
            self.symbol2idx[symbol] = i
            self.abrv2idx[abrv] = i
        self.special_mask = (
            [1 for _ in specials]
            + [0 for _ in aas]
            + [0 for _ in sms]
            + [0 for _ in bases]
        )

        # atom level vocab
        self.idx2atom = [self.atom_pad, self.atom_mask, self.atom_global] + [
            e.upper() for e in chemical_symbols
        ]
        self.idx2atom_pos = [
            self.atom_pos_pad,
            self.atom_pos_mask,
            self.atom_pos_global,
            "",
            "A",
            "B",
            "G",
            "D",
            "E",
            "Z",
            "H",
            "XT",
            "P",
            self.atom_pos_sm,
        ]  # SM is for atoms in small molecule, 'P' for O1P, O2P, O3P
        self.atom2idx, self.atom_pos2idx = {}, {}
        for i, atom in enumerate(self.idx2atom):
            self.atom2idx[atom] = i
        for i, atom_pos in enumerate(self.idx2atom_pos):
            self.atom_pos2idx[atom_pos] = i

    # block level APIs

    def abrv_to_symbol(self, abrv):
        idx = self.abrv_to_idx(abrv)
        return None if idx is None else self.idx2block[idx][0]

    def symbol_to_abrv(self, symbol):
        idx = self.symbol_to_idx(symbol)
        return None if idx is None else self.idx2block[idx][1]

    def abrv_to_idx(self, abrv):
        # abrv = abrv.upper()
        return self.abrv2idx.get(abrv, self.abrv2idx["UNK"])

    def symbol_to_idx(self, symbol):
        # symbol = symbol.upper()
        return self.symbol2idx.get(symbol, self.abrv2idx["UNK"])

    def idx_to_symbol(self, idx):
        return self.idx2block[idx][0]

    def idx_to_abrv(self, idx):
        return self.idx2block[idx][1]

    def get_pad_idx(self):
        return self.symbol_to_idx(self.PAD)

    def get_mask_idx(self):
        return self.symbol_to_idx(self.MASK)

    def get_special_mask(self):
        return copy(self.special_mask)

    # atom level APIs

    def get_atom_pad_idx(self):
        return self.atom2idx[self.atom_pad]

    def get_atom_mask_idx(self):
        return self.atom2idx[self.atom_mask]

    def get_atom_global_idx(self):
        return self.atom2idx[self.atom_global]

    def get_atom_pos_pad_idx(self):
        return self.atom_pos2idx[self.atom_pos_pad]

    def get_atom_pos_mask_idx(self):
        return self.atom_pos2idx[self.atom_pos_mask]

    def get_atom_pos_global_idx(self):
        return self.atom_pos2idx[self.atom_pos_global]

    def idx_to_atom(self, idx):
        return self.idx2atom[idx]

    def atom_to_idx(self, atom):
        atom = atom.upper()
        return self.atom2idx.get(atom, self.atom2idx[self.atom_mask])

    def idx_to_atom_pos(self, idx):
        return self.idx2atom_pos[idx]

    def atom_pos_to_idx(self, atom_pos):
        return self.atom_pos2idx.get(atom_pos, self.atom_pos2idx[self.atom_pos_mask])

    # sizes

    def get_num_atom_type(self):
        return len(self.idx2atom)

    def get_num_atom_pos(self):
        return len(self.idx2atom_pos)

    def get_num_block_type(self):
        return len(self.special_mask) - sum(self.special_mask)

    def is_nuclear_acid(self, idx):
        return len(self.symbol2idx) - idx <= 8

    def __len__(self):
        return len(self.symbol2idx)


VOCAB = MoleculeVocab()
