# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
from typing import Generator, Iterable

import hydra
import torch
from lightning import LightningModule

from nemo.utils import logging


def configure_optimizers(model: LightningModule):
    """
    Re-usable optimizer configuration function for top-level PyTorch Lightning modules in this collection.
    It sets up parameter freezing, optimizer, and LR scheduler.

    The ``model`` object is expected to have a ``model.cfg`` attribute with OmegaConf configuration.
    The following fields are expected:

    * ``optimizer`` with hydra-style ``_target_`` pointing to optimizer class, and the remaining options
        passed directly to its ``__init__`` method.

    * (optional) ``freeze_params`` with a list of regex pattern for identifying frozen parameters.

    * (optional) ``prevent_freeze_params`` with a list of regex pattern for keeping specific parameters trainable
        (overrides ``freeze_params``).

    * (optional) ``param_group_overrides`` with a list of dicts, each containing a ``pattern`` (regex)
        and optimizer kwargs (e.g., ``lr``) to apply to matching parameters.  First matching pattern wins.
        Unmatched parameters inherit the base optimizer settings.  Example::

            param_group_overrides:
              - pattern: "^depthformer\\..*$"
                lr: 1e-4
              - pattern: "^llm\\..*$"
                lr: 1e-5

    * (optional) ``lr_scheduler`` with hydra-style ``_target_`` pointing to LR scheduler class,
        and the remaining options passed directly to its ``__init__`` method.

    Returns:
        PyTorch Lightning Trainer-compatible dict with structure::

            {
                "optimizer": <optimizer>,
                "lr_scheduler": {"scheduler": <lr_scheduler>, "interval": "step", "frequency": 1}
            }

    """
    assert hasattr(model, "cfg"), "Expected `model.cfg` attribute to exist."
    assert "optimizer" in model.cfg, "Expected `model.cfg` to contain 'optimizer' configuration."

    param_group_overrides = list(model.cfg.get("param_group_overrides", []))

    if param_group_overrides:
        named_params = list(freeze_and_subset(
            model.named_parameters(),
            exclude_patterns=model.cfg.get("freeze_params", []),
            keep_patterns=model.cfg.get("prevent_freeze_params", []),
            named=True,
        ))
        param_groups = build_param_groups(named_params, param_group_overrides)
        optimizer = hydra.utils.instantiate(model.cfg.optimizer, param_groups, _convert_='all')
    else:
        parameters = freeze_and_subset(
            model.named_parameters(),
            exclude_patterns=model.cfg.get("freeze_params", []),
            keep_patterns=model.cfg.get("prevent_freeze_params", []),
        )
        optimizer = hydra.utils.instantiate(model.cfg.optimizer, parameters, _convert_='all')

    ans = {"optimizer": optimizer}
    if "lr_scheduler" in model.cfg:
        lr_scheduler = hydra.utils.instantiate(model.cfg.lr_scheduler, optimizer)
        ans["lr_scheduler"] = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
    return ans


def freeze_and_subset(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    exclude_patterns: list[str],
    keep_patterns: list[str] = None,
    named: bool = False,
) -> Generator:
    """
    Utility used to freeze select model parameters, and skip them for the purpose
    of initializing an optimizer's parameter group.

    Args:
        named_parameters: The output of `torch.nn.Module.named_parameters()`
        exclude_patterns: A list of regex patterns matching parameter names to be frozen
            and excluded from optimization.
        keep_patterns: A list of regex patterns matching parameter names to be trained.
            This list overrides all matches to `exclude_patterns`.
        named: If True, yield ``(name, param)`` tuples instead of just ``param``.
            Useful when downstream code needs parameter names (e.g., for param group assignment).

    Returns:
        A generator over parameters (or (name, param) tuples if ``named=True``).

    Example:

        >>> model = MyModel()
        ... # freeze all LLM parameters in "model.llm"
        ... params = freeze_and_subset(model.named_parameters(), [r'^llm\\.\\..+$'])
        ... optimizer = torch.optim.AdamW(params, lr=1e-3)

    """
    exclude_counter = {p: 0 for p in exclude_patterns}

    if not keep_patterns:
        keep_counter = {}

        def _must_keep(_) -> bool:
            return False

    else:
        keep_counter = {p: 0 for p in keep_patterns}
        compiled_keep_patterns = [re.compile(p) for p in keep_patterns]

        def _must_keep(name: str) -> bool:
            for p in compiled_keep_patterns:
                if p.match(name) is not None:
                    keep_counter[p.pattern] += 1
                    return True
            return False

    compiled_exclude_patterns = [re.compile(p) for p in exclude_patterns]

    def _exclude(name: str) -> bool:
        for p in compiled_exclude_patterns:
            if p.match(name) is not None:
                exclude_counter[p.pattern] += 1
                return True
        return False

    trainable, nontrainable = 0, 0
    for name, param in named_parameters:
        discard = False
        if _exclude(name) and not _must_keep(name):
            param.requires_grad = False
            discard = True
        if not discard:
            yield (name, param) if named else param
            trainable += param.numel()
        else:
            nontrainable += param.numel()
    total = trainable + nontrainable

    logging.info(f"Parameters | trainable={trainable} ({trainable / total:.2%}) | total={total}")

    if unused_excluded_patterns := [k for k, v in exclude_counter.items() if v == 0]:
        msg = "['" + "', '".join(unused_excluded_patterns) + "']"
        logging.warning(f"Parameter freezing patterns UNMATCHED against any parameter: {msg} (bad regexp?)")

    if unused_keep_patterns := [k for k, v in keep_counter.items() if v == 0]:
        msg = "['" + "', '".join(unused_keep_patterns) + "']"
        logging.warning(f"Parameter freeze-preventing patterns UNMATCHED against any parameter: {msg} (bad regexp?)")


def build_param_groups(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    overrides: list[dict],
) -> list[dict]:
    """
    Categorize trainable parameters into optimizer param groups based on regex patterns.

    Each override dict must contain a ``pattern`` key (regex) and may contain any optimizer
    kwargs (e.g., ``lr``, ``weight_decay``).  First matching pattern wins.  Parameters that
    don't match any pattern go into a default group that inherits the optimizer's base settings.

    An optional ``name`` key in each override is used for logging and is preserved in the
    resulting param group dict (PyTorch optimizers ignore unknown keys).

    Args:
        named_parameters: List of (name, param) tuples for trainable parameters.
        overrides: List of dicts from ``param_group_overrides`` config.

    Returns:
        List of param group dicts suitable for passing to an optimizer constructor.
    """
    compiled: list[tuple[re.Pattern, str, dict]] = []
    for o in overrides:
        o = dict(o)  # avoid mutating config
        pattern = re.compile(o.pop("pattern"))
        name = o.pop("name", pattern.pattern)
        compiled.append((pattern, name, o))

    # One bucket per override + default
    groups: dict[int, list[torch.nn.Parameter]] = {i: [] for i in range(len(compiled))}
    default_params: list[torch.nn.Parameter] = []

    for pname, param in named_parameters:
        matched = False
        for i, (pattern, _, _) in enumerate(compiled):
            if pattern.search(pname):
                groups[i].append(param)
                matched = True
                break
        if not matched:
            default_params.append(param)

    result: list[dict] = []
    # Default group first (index 0) — no overrides, uses optimizer's base settings.
    if default_params:
        result.append({"params": default_params, "name": "default"})

    for i, (pattern, name, kwargs) in enumerate(compiled):
        params = groups[i]
        if params:
            result.append({"params": params, "name": name, **kwargs})
        else:
            logging.warning(
                f"param_group_overrides pattern '{pattern.pattern}' matched no trainable parameters"
            )

    # Log group stats
    for group in result:
        n_params = sum(p.numel() for p in group["params"])
        extra = ", ".join(f"{k}={v}" for k, v in group.items() if k not in ("params", "name"))
        label = group.get("name", "unnamed")
        msg = f"Param group '{label}': {n_params} parameters"
        if extra:
            msg += f" ({extra})"
        logging.info(msg)

    return result


def is_frozen(module: torch.nn.Module) -> bool:
    return all(not p.requires_grad for p in module.parameters())
