import os
import json
import regex as re
import requests
import torch
from typing import Set, Tuple, Dict, Mapping, Sequence, Any

def bytes_to_unicode():
    safe_byte_values = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range("®"), ord("ÿ")+1))
    safe_unicode_points = safe_byte_values[:]
    n = 0
    for byte in range(2**8):
        if byte not in safe_byte_values
            safe_byte_values.append(byte)
            safe_unicode_points.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in safe_unicode_points]
    byte_to_unicode_map = dict(zip(safe_byte_values, safe_unicode_points))
    return byte_to_unicode_map

def get_pairs(word: str) -> Set[Tuple[str, str]]:
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class Encoder:
    def __init__(self, encoder: Mapping[str, int], bpe_merges: Sequence[Tuple[str, str]]) -> None:
        self.byte_encoder = byte_to_unicode()
        self.byte_decoder = {}
        for byte_value, char in self.byte_encoder.items():
            self.byte_decoder[char] = byte_value
        self.encoder = dict(encoder)
        self.decoder = {}
        for token, token_id in self.encoder.items():
            self.decoder[token_id] = token
        self.bpe_ranks = {}
        for rank, pair in enumerate(bpe_merges):
            self.bpe_ranks[pair] = rank
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.cache = {}

    def bpe(self, token: str):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)
        if not pairs:
            return token
        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word
    
    def encode(self, text: str) -> List[int]:
        encoded_indices = []
        tokens = re.findall(self.pat, text)
        for token in tokens:
            token_bytes = token.encode('utf-8')
            translated_chars = []
            for byte_value in token_bytes:
                translated_char = self.byte_encoder[byte_value]
                translated_chars.append(translated_char)
            token_translated = ''.join(translated_chars)
            merged_token_string = self.bpe(token_translated)
            merged_tokens = merged_token_string.split(' ')
            token_id_list = []
            for subtoken in merged_tokens:
                token_id = self.encoder[subtoken]
                token_id_list.append(token_id)
            for token_id in token_id_list:
                encoded_indices.append(token_id)
        return encoded_indices
    
    def encode_and_show_work(self, text: str) -> Dict[str, Any]:
        encoded_indices = []
        token_debug_info = []
        tokens = re.findall(self.pat, text)
        for token in tokens:
            token_bytes = token.encode("utf-8")
            translated_chars = []
            for byte_value in token_bytes:
                translated_char = self.byte_encoder[byte_value]
                translated_chars.append(translated_char)
            token_translated = "".join(translated_chars)
            merged_token_string = self.bpe(token_translated)
            merged_tokens = merged_token_string.split(" ")
            token_id_list = []
            for subtoken in merged_tokens:
                token_id = self.encoder[subtoken]
                token_id_list.append(token_id)
            for token_id in token_id_list:
                encoded_indices.append(token_id)
            token_debug_info.append({
                'token':token,
                'token_bytes': token_bytes,
                'token_translated': token_translated,
                'token_merged': merged_tokens,
                'token_ids': token_id_list,
            })
        output = {
            'bpe_idx': encoded_indices,
            'tokens': tokens,
            'parts': token_debug_info,
        }
        return output
    
    def decode(self, bpe_idx: List[int]) -> str:
        merged_token_chars = []
        for token_id in bpe_idx:
            bpe_char = self.decoder[token_id]
            merged_token_chars.append(bpe_char)
        merged_text = "".join(merged_token_chars)
        byte_values = []
        for char in merged_text:
            byte_value = self.byte_decoder[char]
            byte_values.append(byte_value)
        raw_bytes = bytearray(byte_values)
        decoded_text = raw_bytes.decode('utf-8', errors = 'replace')
        return decode_text
        
    def get_file(local_fileL str, remote_file: str) -> None:
        if not os.path.isfile(local_file):
            print(f"downloading {remote_file} to {local_file}")
            response = requests.get(remote_file)
            open(local_file, "wb").write(response.content)
    
    def get_encoder() -> Encoder:
        home_dir = os.path.expanduser('~')
        cache_dir = os.path.join(home_dir, '.cache', 'mingpt')
        os.makedirs(cache_dir, exist_ok = True)
        encoder_local_file = os.path.join(cache_dir, 'encoder.json')
        encoder_remote_file = ('https://openaipublic.blob.core.windows.net/'
                               'gpt-2/models/124M/encoder.json')
        get_file(encoder_local_file, encoder_remote_file)
        with open(encoder_local_file, 'r') as file_handle:
            encoder = json.load(file_handle)
        assert len(encoder) == 50257
        vocab_local_file = os.path.join(cache_dir, 'vocab.bpe')
        vocab_remote_file = ('https://openaipublic.blob.core.windows.net/'
                             'gpt-2/models/124M/vocab.bpe')
        get_file(vocab_local_file, vocab_remote_file)
        with open(vocab_local_file, 'r', encoding = 'utf-8') as file_handle:
            bpe_data = file_handle.read()
        bpe_merges = []
        merge_lines = bpe_data.split('\n')[1:-1]
        for merge_line in merge_lines:
            merge_pair = tuple(merge_line.split())
            bpe_merges.append(merge_pair)
        assert len(bpe_merges) == 50000
        encoder_instance = Encoder(encoder, bpe_merges)
        return encoder_instance

class BPETokenizer:
    def __init__(self):
        self.encoder = get_encoder()
    
    def __call__(self, text: str, return_tensors: str = 'pt') -> torch.Tensor:
        assert isinstance(text, str)
        assert return_tensors == 'pt'
        token_ids = self.encoder.encode(text)
        batched_token_ids = [token_ids]
        output_tensor = torch.tensor(batched_token_ids, dtype = torch.long)
        return output_tensor
    
    def decode(self, idx: torch.Tensor) -> str:
        assert isinstance(idx, torch.Tensor)
        assert idx.ndim == 1
        token_id_list = idx.tolist()
        decoded_text = self.encoder.decode(token_id_list)
        return decoded_text