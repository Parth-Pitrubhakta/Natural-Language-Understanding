"""
dataset.py — Character-Level Name Dataset Utilities
====================================================

This module provides all data-handling utilities for character-level name
generation, including:
  - Character vocabulary construction (with special tokens)
  - Encoding / decoding of name strings ↔ integer sequences
  - A PyTorch Dataset wrapper for batched training
  - A collation function for variable-length sequences

Author : Auto-generated for NLU Problem 2
"""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


# ─── Special token constants ────────────────────────────────────────────────
PAD_TOKEN = "<PAD>"   # Padding token (index 0) — used to equalise sequence lengths
SOS_TOKEN = "<SOS>"   # Start-of-sequence token — signals the beginning of a name
EOS_TOKEN = "<EOS>"   # End-of-sequence token   — signals the end of a name


class CharVocab:
    """
    Character-level vocabulary with special tokens.

    Given a list of name strings, this class:
      1. Extracts every unique character that appears in the names.
      2. Prepends three special tokens: <PAD>, <SOS>, <EOS>.
      3. Builds bidirectional mappings  index ↔ character.

    Attributes
    ----------
    itos : list[str]
        Index-to-string mapping  (e.g.  itos[4] → 'A').
    stoi : dict[str, int]
        String-to-index mapping  (e.g.  stoi['A'] → 4).
    """

    def __init__(self, names: list[str]):
        # Collect and sort every character that appears in the training names
        chars = sorted(set("".join(names)))

        # Build the full vocabulary: [PAD, SOS, EOS, char1, char2, …]
        self.special = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]
        self.itos = self.special + chars            # index → character
        self.stoi = {c: i for i, c in enumerate(self.itos)}  # character → index

    # ── Convenience properties for special-token indices ──────────────────
    @property
    def pad_idx(self) -> int:
        """Return the integer index of the <PAD> token."""
        return self.stoi[PAD_TOKEN]

    @property
    def sos_idx(self) -> int:
        """Return the integer index of the <SOS> token."""
        return self.stoi[SOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        """Return the integer index of the <EOS> token."""
        return self.stoi[EOS_TOKEN]

    def __len__(self) -> int:
        """Total vocabulary size (special tokens + unique characters)."""
        return len(self.itos)

    def encode(self, name: str) -> list[int]:
        """
        Convert a name string into a list of integer indices.

        Format: [SOS, c1, c2, …, cn, EOS]

        Parameters
        ----------
        name : str
            The name to encode (e.g. "Aaraa Sethi").

        Returns
        -------
        list[int]
            Integer-encoded sequence including SOS and EOS.
        """
        return [self.sos_idx] + [self.stoi[c] for c in name] + [self.eos_idx]

    def decode(self, indices: list[int]) -> str:
        """
        Convert a list of integer indices back into a name string.

        Stops at the first <EOS> token and strips all special tokens.

        Parameters
        ----------
        indices : list[int]
            Sequence of integer indices from model output.

        Returns
        -------
        str
            Decoded name string.
        """
        chars = []
        for idx in indices:
            tok = self.itos[idx]
            if tok == EOS_TOKEN:
                break
            if tok not in (PAD_TOKEN, SOS_TOKEN):
                chars.append(tok)
        return "".join(chars)


class NameDataset(Dataset):
    """
    PyTorch Dataset that wraps a list of names as encoded integer tensors.

    Each item is a 1-D LongTensor:  [SOS, c1, c2, …, cn, EOS].
    """

    def __init__(self, names: list[str], vocab: CharVocab):
        """
        Parameters
        ----------
        names : list[str]
            Raw name strings.
        vocab : CharVocab
            Vocabulary instance used for encoding.
        """
        self.vocab = vocab
        # Pre-encode every name into a tensor for fast data loading
        self.data = [torch.tensor(vocab.encode(n), dtype=torch.long) for n in names]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def collate_fn(batch: list[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
    """
    Collate function for DataLoader: pads variable-length sequences to
    the length of the longest sequence in the batch.

    Parameters
    ----------
    batch : list[Tensor]
        List of 1-D tensors, each representing an encoded name.
    pad_value : int
        The index used for padding (should be vocab.pad_idx).

    Returns
    -------
    Tensor
        Batch of padded sequences with shape (batch_size, max_seq_len).
    """
    return pad_sequence(batch, batch_first=True, padding_value=pad_value)


def load_names(path: str) -> list[str]:
    """
    Load names from a text file (one name per line).

    Strips whitespace and skips empty lines.

    Parameters
    ----------
    path : str
        Path to the text file.

    Returns
    -------
    list[str]
        List of name strings.
    """
    with open(path, "r") as f:
        names = [line.strip() for line in f if line.strip()]
    return names
