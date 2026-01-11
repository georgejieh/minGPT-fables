import json
import os
import random
import sys
from ast import literal_eval
from typing import Any, Dict, Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Sets RNG seeds for reproducible runs.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(config: "CfgNode") -> None:
    """Creates work dir and writes run snapshots (args + config).

    Args:
        config: Config node. Expects `config.system.work_dir`.
    """
    work_dir = config.system.work_dir
    os.makedirs(work_dir, exist_ok=True)

    args_path = os.path.join(work_dir, "args.txt")
    with open(args_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(" ".join(sys.argv))

    config_path = os.path.join(work_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(json.dumps(config.to_dict(), indent=4))


class CfgNode:
    """Lightweight config object with dotted-key overrides."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)

    def __str__(self) -> str:
        return self._str_helper(indent=0)

    def _str_helper(self, indent: int) -> str:
        lines = []
        for key, value in self.__dict__.items():
            if isinstance(value, CfgNode):
                lines.append((" " * (indent * 4)) + f"{key}:\n")
                lines.append(value._str_helper(indent=indent + 1))
            else:
                lines.append((" " * (indent * 4)) + f"{key}: {value}\n")
        return "".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Converts this config into a plain nested dict.

        Returns:
            Nested dict representation of this config.
        """
        output: Dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, CfgNode):
                output[key] = value.to_dict()
            else:
                output[key] = value
        return output

    def merge_from_dict(self, updates: Dict[str, Any]) -> None:
        """Merges a dict into this config node.

        Args:
            updates: Key/value pairs to set.
        """
        for key, value in updates.items():
            setattr(self, key, value)

    def merge_from_args(self, args: Iterable[str]) -> None:
        """Merges CLI overrides like `--a.b.c=value`.

        Args:
            args: Iterable of CLI override strings.

        Raises:
            AssertionError: If an override is malformed or targets an unknown key.
        """
        for arg in args:
            keyval = arg.split("=", maxsplit=1)
            assert len(keyval) == 2, f"Expected --a.b=value, got: {arg}"

            key, raw_value = keyval
            assert key.startswith("--"), f"Expected --a.b=value, got: {arg}"

            try:
                parsed_value = literal_eval(raw_value)
            except (ValueError, SyntaxError):
                parsed_value = raw_value

            dotted_key = key[2:]
            key_parts = dotted_key.split(".")

            obj: Any = self
            for part in key_parts[:-1]:
                obj = getattr(obj, part)

            leaf_key = key_parts[-1]
            assert hasattr(obj, leaf_key), f"Unknown config key: {dotted_key}"

            print(f"command line overwriting config {dotted_key} with {parsed_value}")
            setattr(obj, leaf_key, parsed_value)
