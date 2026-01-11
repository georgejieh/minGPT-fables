import json
import os
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

import regex as re
import requests
import torch


def bytes_to_unicode() -> Dict[int, str]:
    """Builds a reversible mapping from byte values to unicode characters.

    Returns:
        Mapping from byte values (0-255) to unicode characters.
    """
    safe_byte_values: List[int] = []

    for byte_value in range(ord("!"), ord("~") + 1):
        safe_byte_values.append(byte_value)
    for byte_value in range(ord("¡"), ord("¬") + 1):
        safe_byte_values.append(byte_value)
    for byte_value in range(ord("®"), ord("ÿ") + 1):
        safe_byte_values.append(byte_value)

    safe_unicode_codepoints: List[int] = []
    for byte_value in safe_byte_values:
        safe_unicode_codepoints.append(byte_value)

    extra_codepoint_offset = 0
    for byte_value in range(256):
        if byte_value not in safe_byte_values:
            safe_byte_values.append(byte_value)
            safe_unicode_codepoints.append(256 + extra_codepoint_offset)
            extra_codepoint_offset += 1

    unicode_chars: List[str] = []
    for codepoint in safe_unicode_codepoints:
        unicode_chars.append(chr(codepoint))

    byte_to_unicode_map: Dict[int, str] = {}
    for i in range(len(safe_byte_values)):
        byte_value = safe_byte_values[i]
        unicode_char = unicode_chars[i]
        byte_to_unicode_map[byte_value] = unicode_char

    return byte_to_unicode_map


def get_pairs(symbols: Tuple[str, ...]) -> Set[Tuple[str, str]]:
    """Returns adjacent symbol pairs in a token.

    Args:
        symbols: Token as a tuple of symbols.

    Returns:
        Set of consecutive symbol pairs.
    """
    pairs: Set[Tuple[str, str]] = set()
    if len(symbols) < 2:
        return pairs

    prev_symbol = symbols[0]
    for symbol in symbols[1:]:
        pairs.add((prev_symbol, symbol))
        prev_symbol = symbol

    return pairs


def get_file(local_file: str, remote_file: str) -> None:
    """Downloads a file if it does not exist locally.

    Args:
        local_file: Destination path.
        remote_file: Source URL.
    """
    if os.path.isfile(local_file):
        return

    print(f"downloading {remote_file} to {local_file}")
    response = requests.get(remote_file, timeout=60)
    response.raise_for_status()

    with open(local_file, "wb") as file_handle:
        file_handle.write(response.content)


def get_encoder() -> "Encoder":
    """Loads GPT-2 BPE assets and returns an Encoder."""
    home_dir = os.path.expanduser("~")
    cache_dir = os.path.join(home_dir, ".cache", "mingpt_fables")
    os.makedirs(cache_dir, exist_ok=True)

    encoder_local_file = os.path.join(cache_dir, "encoder.json")
    encoder_remote_file = (
        "https://openaipublic.blob.core.windows.net/"
        "gpt-2/models/124M/encoder.json"
    )
    get_file(encoder_local_file, encoder_remote_file)

    with open(encoder_local_file, "r", encoding="utf-8") as file_handle:
        encoder = json.load(file_handle)

    # GPT-2 vocab size.
    assert len(encoder) == 50257

    vocab_local_file = os.path.join(cache_dir, "vocab.bpe")
    vocab_remote_file = (
        "https://openaipublic.blob.core.windows.net/"
        "gpt-2/models/124M/vocab.bpe"
    )
    get_file(vocab_local_file, vocab_remote_file)

    with open(vocab_local_file, "r", encoding="utf-8") as file_handle:
        bpe_data = file_handle.read()

    bpe_merges: List[Tuple[str, str]] = []
    merge_lines = bpe_data.split("\n")[1:-1]
    for merge_line in merge_lines:
        parts = merge_line.split()
        if len(parts) != 2:
            continue
        bpe_merges.append((parts[0], parts[1]))

    # GPT-2 merge count.
    assert len(bpe_merges) == 50000

    return Encoder(encoder=encoder, bpe_merges=bpe_merges)


class Encoder:
    """GPT-2 style BPE encoder/decoder."""

    def __init__(
        self,
        encoder: Mapping[str, int],
        bpe_merges: Sequence[Tuple[str, str]],
    ) -> None:
        """Initializes encoder.

        Args:
            encoder: Mapping from token string to token ID.
            bpe_merges: Ordered list of merge pairs.
        """
        self.byte_encoder = bytes_to_unicode()

        self.byte_decoder: Dict[str, int] = {}
        for byte_value, unicode_char in self.byte_encoder.items():
            self.byte_decoder[unicode_char] = byte_value

        self.encoder: Dict[str, int] = dict(encoder)

        self.decoder: Dict[int, str] = {}
        for token, token_id in self.encoder.items():
            self.decoder[token_id] = token

        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        for rank, pair in enumerate(bpe_merges):
            self.bpe_ranks[pair] = rank

        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| """
            r"""?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        self.cache: Dict[str, str] = {}

    def bpe(self, token: str) -> str:
        """Applies BPE merges to a single token.

        Args:
            token: Token string after byte-to-unicode translation.

        Returns:
            A space-separated string of merged symbols.
        """
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            self.cache[token] = token
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break

            first_symbol, second_symbol = bigram
            new_word: List[str] = []
            i = 0

            while i < len(word):
                try:
                    j = word.index(first_symbol, i)
                except ValueError:
                    for tail_symbol in word[i:]:
                        new_word.append(tail_symbol)
                    break

                for symbol in word[i:j]:
                    new_word.append(symbol)

                i = j

                if i < len(word) - 1 and word[i] == first_symbol and word[i + 1] == second_symbol:
                    new_word.append(first_symbol + second_symbol)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            if len(word) == 1:
                break

            pairs = get_pairs(word)

        merged = " ".join(word)
        self.cache[token] = merged
        return merged

    def encode(self, text: str) -> List[int]:
        """Encodes text into token IDs.

        Args:
            text: Input text.

        Returns:
            Token IDs.
        """
        assert isinstance(text, str)

        encoded_ids: List[int] = []
        tokens = re.findall(self.pat, text)

        for token in tokens:
            token_bytes = token.encode("utf-8")

            translated_chars: List[str] = []
            for byte_value in token_bytes:
                translated_chars.append(self.byte_encoder[byte_value])

            translated_token = "".join(translated_chars)

            merged_token_string = self.bpe(translated_token)
            merged_tokens = merged_token_string.split(" ")

            token_ids: List[int] = []
            for subtoken in merged_tokens:
                token_ids.append(self.encoder[subtoken])

            for token_id in token_ids:
                encoded_ids.append(token_id)

        return encoded_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decodes token IDs into text.

        Args:
            token_ids: Token IDs.

        Returns:
            Decoded text.
        """
        merged_token_chars: List[str] = []
        for token_id in token_ids:
            merged_token_chars.append(self.decoder[token_id])

        merged_text = "".join(merged_token_chars)

        byte_values: List[int] = []
        for char in merged_text:
            byte_values.append(self.byte_decoder[char])

        raw_bytes = bytearray(byte_values)
        decoded_text = raw_bytes.decode("utf-8", errors="replace")
        return decoded_text


class BPETokenizer:
    """Tokenizer wrapper that returns torch tensors."""

    def __init__(self) -> None:
        self.encoder = get_encoder()

    def __call__(self, text: str, return_tensors: str = "pt") -> torch.Tensor:
        """Encodes text and returns a tensor with batch dimension.

        Args:
            text: Input text.
            return_tensors: Must be "pt".

        Returns:
            Token ID tensor of shape (1, T).
        """
        assert isinstance(text, str)
        assert return_tensors == "pt"

        token_ids = self.encoder.encode(text)
        batch: List[List[int]] = [token_ids]
        output_tensor = torch.tensor(batch, dtype=torch.long)
        return output_tensor

    def decode(self, idx: torch.Tensor) -> str:
        """Decodes a 1D tensor of token IDs.

        Args:
            idx: 1D tensor of token IDs.

        Returns:
            Decoded text.
        """
        assert isinstance(idx, torch.Tensor)
        assert idx.ndim == 1

        token_ids = idx.tolist()
        decoded_text = self.encoder.decode(token_ids)
        return decoded_text
