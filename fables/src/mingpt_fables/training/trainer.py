import time
from collections import defaultdict
from typing import Callable, DefaultDict, List, Optional

import torch
from torch.utils.data import DataLoader, RandomSampler

from mingpt_fables.utils.misc import CfgNode


Callback = Callable[["Trainer"], None]


class Trainer:
    """Simple supervised trainer (next-token prediction).

    RL fine-tuning belongs in `mingpt_fables/rl` so the concerns don't mix.
    """

    @staticmethod
    def get_default_config() -> CfgNode:
        """Returns default training config."""
        config = CfgNode()
        config.device = "auto"
        config.num_workers = 4

        config.max_iters = None
        config.batch_size = 64

        config.learning_rate = 3e-4
        config.betas = (0.9, 0.95)
        config.weight_decay = 0.1
        config.grad_norm_clip = 1.0
        return config

    def __init__(
        self,
        config: CfgNode,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
    ) -> None:
        """Initializes trainer.

        Args:
            config: Training config.
            model: Language model to train.
            train_dataset: Dataset yielding (input_ids, targets).
        """
        self.config = config
        self.model = model
        self.train_dataset = train_dataset

        self.callbacks: DefaultDict[str, List[Callback]] = defaultdict(list)
        self.optimizer: Optional[torch.optim.Optimizer] = None

        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device

        self.model = self.model.to(self.device)
        print("running on device", self.device)

        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0
        self.loss: Optional[torch.Tensor] = None

    def add_callback(self, on_event: str, callback: Callback) -> None:
        """Adds a callback to an event."""
        self.callbacks[on_event].append(callback)

    def set_callback(self, on_event: str, callback: Callback) -> None:
        """Replaces callbacks for an event with a single callback."""
        self.callbacks[on_event] = [callback]

    def trigger_callbacks(self, on_event: str) -> None:
        """Triggers callbacks for an event."""
        for callback in self.callbacks.get(on_event, []):
            callback(self)

    def run(self) -> None:
        """Runs training loop until `max_iters` is reached."""
        model = self.model
        config = self.config

        self.optimizer = model.configure_optimizers(config)

        sampler = RandomSampler(
            self.train_dataset,
            replacement=True,
            num_samples=int(1e10),
        )

        train_loader = DataLoader(
            self.train_dataset,
            sampler=sampler,
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)

        while True:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            batch_tensors: List[torch.Tensor] = []
            for tensor in batch:
                batch_tensors.append(tensor.to(self.device))

            input_ids, targets = batch_tensors

            _, loss = model(input_ids, targets)
            self.loss = loss

            model.zero_grad(set_to_none=True)
            assert loss is not None
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.grad_norm_clip,
            )

            assert self.optimizer is not None
            self.optimizer.step()

            self.trigger_callbacks("on_batch_end")

            self.iter_num += 1
            now = time.time()
            self.iter_dt = now - self.iter_time
            self.iter_time = now

            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
