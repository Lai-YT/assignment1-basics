import heapq
import multiprocessing
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from functools import partial, total_ordering
from itertools import pairwise
from typing import cast

import regex as re

from cs336_basics.pretokenization_example import find_chunk_boundaries

END_OF_TEXT = b"<|endoftext|>"
GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
try:
    # Returns the number of CPUs the current process is restricted to.
    available_cores = len(os.sched_getaffinity(0))
except AttributeError:
    # Fallback for Windows/macOS where sched_getaffinity isn't available.
    available_cores = multiprocessing.cpu_count()
# Leave 2 cores to other applications.
NUM_PROCESSES = max(1, available_cores - 2)


def pretokenize(
    boundary: tuple[int, int], input_path: str, special_tokens: list[str]
) -> list[list[int]]:
    """Pretokenize on file range [`boundary[0]`, `boundary[1]`)."""
    start, end = boundary
    indices = list[list[int]]()
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token.
    for split in re.split("|".join(re.escape(st) for st in special_tokens), chunk):
        for m in re.finditer(GPT2_PATTERN, split):
            indices.append(list(map(int, m.group().encode())))
    return indices


@total_ordering
class ReverseOrder:
    def __init__(self, value):
        self._value = value

    def __eq__(self, other):
        return self._value == other._value

    def __lt__(self, other):
        return self._value > other._value


class PriorityQueue[T]:
    def __init__(self, sentinel: T):
        self._sentinel = sentinel
        self._pq = []
        self._entry_finder = {}

    def add(self, priority, tie_breaker, task: T):
        if task in self._entry_finder:
            self._remove(task)
        entry = [ReverseOrder(priority), ReverseOrder(tie_breaker), task]
        self._entry_finder[task] = entry
        heapq.heappush(self._pq, entry)

    def _remove(self, task):
        entry = self._entry_finder.pop(task)
        entry[-1] = self._sentinel

    def pop(self) -> T | None:
        while self._pq:
            _, _, task = heapq.heappop(self._pq)
            if task != self._sentinel:
                del self._entry_finder[task]
                return cast(T, task)


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Args:
        input_path: Path to a text file with BPE tokenizer training data.
        vocab_size: A positive integer that defines the maximum final vocabulary size
            (including the initial byte vocabulary, vocabulary items produced from merging,
            and any special tokens).
        special_tokens: A list of strings to add to the vocabulary.
            During training, treat them as hard boundaries that prevent merges across their spans,
            but do not include them when computing merge statistics.

    Returns:
        vocab: A mapping from int (token ID in the vocabulary) to bytes (token bytes).
        merges: A list of BPE merges produced from training. Each list item is a tuple of
            bytes `(<token1>, <token2>)`, representing that `<token1>` was merged with `<token2>`.
            The merges should be ordered by order of creation.
    """
    if END_OF_TEXT.decode() not in special_tokens:
        special_tokens += [END_OF_TEXT.decode()]
    # pre-token id -> indices
    indices = list[list[int]]()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, NUM_PROCESSES, END_OF_TEXT)
        # # Serial implementation:
        # for boundary in pairwise(boundaries):
        #     indices.extend(pretokenize(boundary, input_path, special_tokens))
    pretokenize_ = partial(pretokenize, input_path=input_path, special_tokens=special_tokens)
    with multiprocessing.Pool(NUM_PROCESSES) as pool:
        for chunk_indices in pool.imap_unordered(pretokenize_, list(pairwise(boundaries))):
            indices.extend(chunk_indices)

    NUM_BYTE_VALUES = 256
    # index -> bytes
    vocab = {x: bytes([x]) for x in range(NUM_BYTE_VALUES)}
    vocab |= {x + NUM_BYTE_VALUES: special_tokens[x].encode() for x in range(len(special_tokens))}
    vocab_count = NUM_BYTE_VALUES + len(special_tokens)
    # index1, index2 -> merged index
    merges = dict[tuple[int, int], int]()
    # Caching for faster merging; a sequence may contribute multiple counts.
    # index1, index2 -> sequene index
    occurences = defaultdict[tuple[int, int], set[int]](set)
    # index1, index2 -> count
    counts = Counter[tuple[int, int]]()
    pq = PriorityQueue[tuple[int, int]](sentinel=(-1, -1))
    for seq_idx, seq in enumerate(indices):
        # Count the number of occurrences of each pair of tokens.
        for pair in pairwise(seq):
            counts[pair] += 1
            occurences[pair].add(seq_idx)
    for p, c in counts.items():
        # Tie break by lexicographical order.
        pq.add(c, tuple(map(vocab.get, p)), p)

    # In each round, a merge is made.
    for _ in range(vocab_size - vocab_count):
        # Find the most common pair.
        # pair = max(counts, key=lambda p: (counts[p], (vocab[p[0]], vocab[p[1]])))  # type: ignore
        pair = pq.pop()
        if pair is None:
            # All pairs are merged.
            break
        index1, index2 = pair
        # Merge that pair.
        new_index = vocab_count
        vocab_count += 1
        merges[pair] = new_index
        vocab[new_index] = vocab[index1] + vocab[index2]
        del counts[pair]

        # Recompute the count of the affected sequences along the merge.
        affected = list(occurences.pop(pair))
        changes = Counter[tuple[int, int]]()
        # NOTE: The overhead of distributing to multiple processes out-weights the benefit.
        for seq_idx in affected:
            new_indices = list[int]()
            old_indices = indices[seq_idx]
            n = len(old_indices)
            i = 0
            while i + 1 < n:
                a, b = old_indices[i], old_indices[i + 1]
                if (a, b) == pair:
                    # The merged pair is not included in the change because its entry is removed entirely.
                    new_indices.append(new_index)
                    if i != 0:
                        old_pair = (old_indices[i - 1], a)
                        new_pair = (old_indices[i - 1], new_index)
                        changes[old_pair] -= 1
                        changes[new_pair] += 1
                        occurences[new_pair].add(seq_idx)
                    if i + 2 < n:
                        old_pair = (b, old_indices[i + 2])
                        new_pair = (new_index, old_indices[i + 2])
                        changes[old_pair] -= 1
                        changes[new_pair] += 1
                        occurences[new_pair].add(seq_idx)
                    i += 2
                else:
                    new_indices.append(old_indices[i])
                    i += 1
            if i < n:
                new_indices.append(old_indices[i])
            indices[seq_idx] = new_indices
        for p, c in changes.items():
            counts[p] += c
            pq.add(counts[p], tuple(map(vocab.get, p)), p)
    return vocab, [(vocab[index1], vocab[index2]) for index1, index2 in merges.keys()]


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_token: list[str] | None = None,
    ) -> None:
        """Construct a tokenizer from a given vocabulary, list of merges,
        and (optionally) a list of special tokens."""
        raise NotImplementedError

    @classmethod
    def from_files(
        cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None
    ) -> "BPETokenizer":
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        raise NotImplementedError

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.

        This is required for memory-efficient tokenization of large files that we cannot directly load into memory."""
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        raise NotImplementedError
