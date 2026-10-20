"""BPE tokenizer: encoding and decoding (assignment 1, section 2.6)."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
import regex as re
import json

_PRINTABLE_BYTES = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))


def _build_byte_to_unicode() -> dict[int, str]:
    printable = set(_PRINTABLE_BYTES)
    mapping: dict[int, str] = {}
    n = 0
    for b in range(256):
        if b in printable:
            mapping[b] = chr(b)
        else:
            mapping[b] = chr(256 + n)
            n += 1
    return mapping


BYTE_TO_UNICODE = _build_byte_to_unicode()

UNICODE_TO_BYTE = {v: k for k, v in BYTE_TO_UNICODE.items()}

def _str_to_merges(string: str) -> bytes:
    """Convert a string to bytes using the BYTE_TO_UNICODE mapping."""
    return bytes(UNICODE_TO_BYTE[c] for c in string)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.byte_encoder = {v: k for k, v in vocab.items()}
        self.merges_rank = {v: i for i, v in enumerate(merges)}
        self.PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        if special_tokens is not None:
            
            for token in special_tokens:
                token_bytes = token.encode('utf-8')
                if token_bytes not in self.byte_encoder:
                    self.byte_encoder[token_bytes] = len(self.byte_encoder)
            special_sorts = sorted(special_tokens, key=lambda x: len(x), reverse=True)
            self.special_pat = re.compile('(' + '|'.join(re.escape(token) for token in special_sorts)+')')

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_filepath) as f:
            vocab = json.load(f)
            vocab = {int(v): _str_to_merges(k) for k, v in vocab.items()}  # Ensure keys are integers
        with open(merges_filepath) as f:
            merges = []
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    merges.append(tuple(_str_to_merges(c) for c in cleaned_line.split(" ")))

        return cls(vocab, merges, special_tokens=special_tokens)

    def _encode_word(self, word: str) -> list[int]:
        word_bytes = [bytes([_]) for _ in word.encode('utf-8')]
        merge_items = dict()
        for i in range(len(word_bytes) - 1):
            pair = (word_bytes[i], word_bytes[i + 1])
            if pair in self.merges_rank:
                merge_items[pair] = self.merges_rank[pair]
        while len(merge_items) > 0:
            best_pair = min(merge_items, key=merge_items.get)
            first, second = best_pair
            new_word_bytes = []
            i = 0
            while i < len(word_bytes):
                if i < len(word_bytes) - 1 and word_bytes[i] == first and word_bytes[i + 1] == second:
                    new_word_bytes.append(first + second)
                    i += 2
                else:
                    new_word_bytes.append(word_bytes[i])
                    i += 1
            word_bytes = new_word_bytes
            merge_items.clear()
            for i in range(len(word_bytes) - 1):
                pair = (word_bytes[i], word_bytes[i + 1])
                if pair in self.merges_rank:
                    merge_items[pair] = self.merges_rank[pair]
        
        return [self.byte_encoder[b] for b in word_bytes]


    def encode(self, text: str) -> list[int]:
        res = []
        if self.special_tokens is None:
            words = self.PAT.findall(text)
            for word in words:
                res.extend(self._encode_word(word))
            
        else:
            for txt in re.split(self.special_pat, text):
                if txt in self.special_tokens:
                    res.append(self.byte_encoder[txt.encode('utf-8')])
                    continue

                words = self.PAT.findall(txt)
                for word in words:
                    res.extend(self._encode_word(word))
                # encoder words
        return res

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        prefix = ""
        for text in iterable:
            text = prefix + text
            if self.special_tokens is None:
                words = self.PAT.findall(text)
                for word in words[:-1]:
                    yield from self._encode_word(word)
                prefix = words[-1] if words else ""
                
            else:
                txts = re.split(self.special_pat, text)
                for txt in txts[:-1]:
                    if txt in self.special_tokens:
                        yield self.byte_encoder[txt.encode('utf-8')]
                        continue

                    words = self.PAT.findall(txt)
                    for word in words:
                        yield from self._encode_word(word)
                if txts[-1] in self.special_tokens:
                    yield self.byte_encoder[txts[-1].encode('utf-8')]
                    prefix = ""
                else:
                    words = self.PAT.findall(txts[-1])
                    for word in words[:-1]:
                        yield from self._encode_word(word)
                    prefix = words[-1] if words else ""
                
        if prefix:
            for word in self.PAT.findall(prefix):
                yield from self._encode_word(word)
        


            

    def decode(self, ids: list[int]) -> str:
        bytes_list = [self.vocab[i] for i in ids]
        return b''.join(bytes_list).decode('utf-8', errors='ignore')
        