import json
import os
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

import regex as re
import requests
import torch

def bytes_to_unicode() -> Dict[int, str]:
    """Build a reversible mapping from byte values to unicode characters.

    Returns:
        A mapping from byte values (0-255) to unicode characters used by the
        GPT-2 byte fallback scheme.
    """
    safe_byte_values = []
    for byte_value in range(ord("!"), ord("~") + 1):
        safe_byte_values.append(byte_value)
    for byte_value in range(ord("¡"), ord("¬") + 1):
        safe_byte_values.append(byte_value)
    for byte_value in range(ord("®"), ord("ÿ") + 1):
        safe_byte_values.append(byte_value)

    safe_unicode_codepoints = []
    for byte_value in safe_byte_values:
        safe_unicode_codepoints.append(byte_value)

    extra_codepoint_offset = 0
    for byte_value in range(256):
        if byte_value not in safe_byte_values:
            safe_byte_values.append(byte_value)
            safe_unicode_codepoints.append(256 + extra_codepoint_offset)
            extra_codepoint_offset += 1

    unicode_chars = []
    for codepoint in safe_unicode_codepoints:
        unicode_chars.append(chr(codepoint))

    byte_to_unicode_map = {}
    for byte_value, unicode_char in zip(safe_byte_values, unicode_chars):
        byte_to_unicode_map[byte_value] = unicode_char

    return byte_to_unicode_map

def get_pairs(symbols: Tuple[str, ...]) -> Set[Tuple[str, str]]:
    """Return adjacent symbol pairs in a token.

    Args:
        symbols: A token represented as a tuple of string symbols.

    Returns:
        A set of (symbol_i, symbol_{i+1}) pairs.
    """
    if len(symbols) < 2:
        return set()

    pairs: Set[Tuple[str, str]] = set()
    previous_symbol = symbols[0]

    for symbol in symbols[1:]:
        pairs.add((previous_symbol, symbol))
        previous_symbol = symbol

    return pairs

def get_file(local_file: str, remote_file: str) -> None:
    """Download a remote file to disk if it does not already exist.

    Args:
        local_file: Destination path on the local filesystem.
        remote_file: URL of the remote file to download.
    """
    if os.path.isfile(local_file):
        return

    print(f"downloading {remote_file} to {local_file}")
    response = requests.get(remote_file, timeout=60)
    response.raise_for_status()

    with open(local_file, "wb") as file_handle:
        file_handle.write(response.content)

def get_encoder() -> "Encoder":
    """Load GPT-2 tokenizer assets and return an initialized Encoder.

    Returns:
        An Encoder initialized with GPT-2's encoder.json and vocab.bpe merges.
    """
    home_dir = os.path.expanduser("~")
    cache_dir = os.path.join(home_dir, ".cache", "mingpt")
    os.makedirs(cache_dir, exist_ok=True)

    encoder_local_file = os.path.join(cache_dir, "encoder.json")
    encoder_remote_file = (
        "https://openaipublic.blob.core.windows.net/"
        "gpt-2/models/124M/encoder.json"
    )
    get_file(encoder_local_file, encoder_remote_file)

    with open(encoder_local_file, "r", encoding="utf-8") as file_handle:
        encoder = json.load(file_handle)

    # GPT-2 tokenizer vocab size.
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

    # GPT-2 BPE merge rules count.
    assert len(bpe_merges) == 50000

    return Encoder(encoder=encoder, bpe_merges=bpe_merges)

class Encoder:
    """GPT-2 style BPE encoder/decoder."""

    def __init__(
        self,
        encoder: Mapping[str, int],
        bpe_merges: Sequence[Tuple[str, str]],
    ) -> None:
        """Initialize an Encoder with token IDs and BPE merge rules.

        Args:
            encoder: Mapping from token strings to integer token IDs.
            bpe_merges: Ordered sequence of BPE merge pairs.
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
        """Apply BPE merges to a single token.

        Args:
            token: A token string to be merged.

        Returns:
            A space-delimited string of merged symbols.
        """
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            self.cache[token] = token
            return token

        while True:
            best_pair = min(
                pairs,
                key=lambda pair: self.bpe_ranks.get(pair, float("inf")),
            )

            if best_pair not in self.bpe_ranks:
                break

            first, second = best_pair
            new_word: List[str] = []
            i = 0

            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    for tail_symbol in word[i:]:
                        new_word.append(tail_symbol)
                    break

                for symbol in word[i:j]:
                    new_word.append(symbol)

                i = j

                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
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
        """Encode a string into a flat list of token IDs.

        Args:
            text: Input text to encode.

        Returns:
            A flat list of integer token IDs.
        """
        assert isinstance(text, str)

        encoded_indices: List[int] = []
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
                encoded_indices.append(token_id)

        return encoded_indices

    def encode_and_show_work(self, text: str) -> Dict[str, Any]:
        """Encode text and return intermediate states for debugging.

        Args:
            text: Input text to encode.

        Returns:
            A dictionary containing token IDs and intermediate tokenization steps.
        """
        assert isinstance(text, str)

        encoded_indices: List[int] = []
        token_debug_info: List[Dict[str, Any]] = []
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
                encoded_indices.append(token_id)

            token_debug_info.append(
                {
                    "token": token,
                    "token_bytes": token_bytes,
                    "token_translated": translated_token,
                    "token_merged": merged_tokens,
                    "token_ids": token_ids,
                }
            )

        return {
            "bpe_idx": encoded_indices,
            "tokens": tokens,
            "parts": token_debug_info,
        }

    def decode(self, bpe_idx: List[int]) -> str:
        """Decode a flat list of token IDs into a UTF-8 string.

        Args:
            bpe_idx: A flat list of integer token IDs.

        Returns:
            The decoded UTF-8 string.
        """
        merged_token_chars: List[str] = []
        for token_id in bpe_idx:
            merged_token_chars.append(self.decoder[token_id])

        merged_text = "".join(merged_token_chars)

        byte_values: List[int] = []
        for char in merged_text:
            byte_values.append(self.byte_decoder[char])

        raw_bytes = bytearray(byte_values)
        decoded_text = raw_bytes.decode("utf-8", errors="replace")
        return decoded_text

class BPETokenizer:
    """Wrap an Encoder and emit PyTorch tensors for model inputs."""

    def __init__(self) -> None:
        """Initialize the tokenizer by loading GPT-2 tokenizer assets."""
        self.encoder = get_encoder()

    def __call__(self, text: str, return_tensors: str = "pt") -> torch.Tensor:
        """Encode text and return a PyTorch tensor with a batch dimension.

        Args:
            text: Input string to encode.
            return_tensors: Output tensor type. Only "pt" is supported.

        Returns:
            A torch.LongTensor of shape (1, seq_len).
        """
        assert isinstance(text, str)
        assert return_tensors == "pt"

        token_ids = self.encoder.encode(text)

        batched_token_ids: List[List[int]] = []
        batched_token_ids.append(token_ids)

        output_tensor = torch.tensor(batched_token_ids, dtype=torch.long)
        return output_tensor

    def decode(self, idx: torch.Tensor) -> str:
        """Decode a 1D tensor of token IDs into a string.

        Args:
            idx: A 1D tensor of token IDs.

        Returns:
            The decoded string.
        """
        assert isinstance(idx, torch.Tensor)
        assert idx.ndim == 1

        token_id_list = idx.tolist()
        decoded_text = self.encoder.decode(token_id_list)
        return decoded_text