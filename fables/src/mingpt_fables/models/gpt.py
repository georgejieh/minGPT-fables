import math
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from mingpt_fables.utils.misc import CfgNode


class NewGELU(nn.Module):
    """GELU used by GPT-2."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (
            1.0
            + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
            )
        )


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config: CfgNode) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        bias = torch.tril(torch.ones(config.block_size, config.block_size))
        bias = bias.view(1, 1, config.block_size, config.block_size)
        self.register_buffer("bias", bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    """Transformer block."""

    def __init__(self, config: CfgNode) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)

        self.mlp_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.mlp_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.mlp_act = NewGELU()
        self.mlp_dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))

        hidden = self.mlp_fc(self.ln_2(x))
        hidden = self.mlp_act(hidden)
        hidden = self.mlp_proj(hidden)
        hidden = self.mlp_dropout(hidden)

        x = x + hidden
        return x


class GPT(nn.Module):
    """GPT language model."""

    @staticmethod
    def get_default_config() -> CfgNode:
        """Returns default model config."""
        config = CfgNode()
        config.model_type = "gpt"

        config.n_layer = None
        config.n_head = None
        config.n_embd = None

        config.vocab_size = None
        config.block_size = None

        config.embd_pdrop = 0.1
        config.resid_pdrop = 0.1
        config.attn_pdrop = 0.1
        return config

    def __init__(self, config: CfgNode) -> None:
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        self.block_size = config.block_size

        type_given = config.model_type is not None
        params_given = (
            config.n_layer is not None
            and config.n_head is not None
            and config.n_embd is not None
        )
        assert type_given ^ params_given

        if type_given:
            presets: Dict[str, Dict[str, int]] = {
                "gpt-nano": {"n_layer": 3, "n_head": 3, "n_embd": 48},
                "gpt-micro": {"n_layer": 4, "n_head": 4, "n_embd": 128},
                "gpt-mini": {"n_layer": 6, "n_head": 6, "n_embd": 192},
            }
            assert config.model_type in presets, f"Unknown model_type: {config.model_type}"
            config.merge_from_dict(presets[config.model_type])

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

        blocks = []
        for _ in range(config.n_layer):
            blocks.append(Block(config))
        self.h = nn.ModuleList(blocks)

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                # Matches GPT-2 init. The scaling depends on depth, so it can look random.
                torch.nn.init.normal_(
                    param,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * config.n_layer),
                )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            return

        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            return

        if isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
            return

    def configure_optimizers(self, train_config: CfgNode) -> torch.optim.Optimizer:
        """Creates AdamW optimizer with weight-decay grouping.

        Args:
            train_config: Training config with lr/betas/weight_decay.

        Returns:
            AdamW optimizer.
        """
        decay_names: Set[str] = set()
        no_decay_names: Set[str] = set()

        whitelist_modules = (torch.nn.Linear,)
        blacklist_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for module_name, module in self.named_modules():
            for param_name, _ in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{param_name}" if module_name else param_name

                if param_name.endswith("bias"):
                    no_decay_names.add(full_name)
                    continue

                if not param_name.endswith("weight"):
                    continue

                if isinstance(module, whitelist_modules):
                    decay_names.add(full_name)
                    continue

                if isinstance(module, blacklist_modules):
                    no_decay_names.add(full_name)
                    continue

        param_dict: Dict[str, torch.nn.Parameter] = {}
        for name, param in self.named_parameters():
            param_dict[name] = param

        overlap = decay_names & no_decay_names
        assert not overlap, f"params in both decay and no_decay: {overlap}"

        unassigned = set(param_dict.keys()) - (decay_names | no_decay_names)
        assert not unassigned, f"params not assigned to decay/no_decay: {unassigned}"

        decay_params: List[torch.nn.Parameter] = []
        no_decay_params: List[torch.nn.Parameter] = []

        # Keep deterministic ordering without writing our own sort logic.
        for name in sorted(param_dict.keys()):
            param = param_dict[name]
            if name in decay_names:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": train_config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=train_config.learning_rate,
            betas=train_config.betas,
        )
        return optimizer

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Runs a forward pass.

        Args:
            input_ids: Token IDs of shape (B, T).
            targets: Optional target IDs of shape (B, T).

        Returns:
            (logits, loss). Loss is None if targets is None.
        """
        device = input_ids.device
        _, seq_len = input_ids.size()

        assert seq_len <= self.block_size, (
            f"seq_len={seq_len} exceeds block_size={self.block_size}"
        )

        pos = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.wte(input_ids)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        for block in self.h:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    def log_probs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns per-token log-probs. Handy for RL rollouts."""
        logits, _ = self(input_ids, targets=None)
        return F.log_softmax(logits, dim=-1)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generates tokens autoregressively.

        Args:
            input_ids: Input token IDs of shape (B, T).
            max_new_tokens: Number of new tokens to append.
            temperature: Softmax temperature.
            do_sample: If True, sample from the distribution.
            top_k: If set, restrict sampling to top-k tokens.

        Returns:
            Token IDs with generated continuation appended.
        """
        idx = input_ids

        for _ in range(max_new_tokens):
            idx_cond = idx
            if idx.size(1) > self.block_size:
                idx_cond = idx[:, -self.block_size :]

            logits, _ = self(idx_cond, targets=None)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                cutoff = values[:, [-1]]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))

            probs = F.softmax(logits, dim=-1)

            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx
