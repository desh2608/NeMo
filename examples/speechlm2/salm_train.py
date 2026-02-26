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
import os

import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

from nemo.collections.speechlm2 import SALM, DataModule, SALMDataset
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

if torch.cuda.is_available():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


@hydra_runner(config_path="conf", config_name="salm")
def train(cfg):
    OmegaConf.resolve(cfg)
    if torch.cuda.is_available():
        torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("medium")
    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    OmegaConf.save(cfg, log_dir / "exp_config.yaml")

    with trainer.init_module():
        model = SALM(OmegaConf.to_container(cfg.model, resolve=True))

    dataset_kwargs = {}
    if cfg.model.get("depthformer") is not None:
        dataset_kwargs["audio_locator_tag"] = cfg.model.audio_locator_tag
        dataset_kwargs["audio_out_locator_tag"] = cfg.model.get("audio_out_locator_tag", "<|audio_out|>")
        dataset_kwargs["audio_start_tag"] = cfg.model.get("audio_start_tag", "<|audio_start|>")
    if cfg.model.get("context_audio_locator_tag") is not None:
        dataset_kwargs["context_audio_locator_tag"] = cfg.model.context_audio_locator_tag
    dataset = SALMDataset(tokenizer=model.tokenizer, **dataset_kwargs)
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
