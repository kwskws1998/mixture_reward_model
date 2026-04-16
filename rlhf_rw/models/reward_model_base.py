import sys
import os

from utils.lmdb_storage import LMDBStorage
import pathlib
import hashlib

sys.path.append("../..")
import numpy as np

path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve().parent.resolve())
sys.path.append(path)
path = str(
    pathlib.Path(__file__)
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
)
sys.path.append(path)
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from eyetrackpy.data_generator.fixations_predictor_trained_1.fixations_predictor_model_1 import (
    FixationsPredictor_1,
)

try:
    from et2_wrapper import FixationsPredictor_2
except ImportError:
    import importlib.util, pathlib as _pl
    _spec = importlib.util.spec_from_file_location(
        "et2_wrapper",
        str(_pl.Path(__file__).parent.parent.parent / "et2_wrapper.py"),
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    FixationsPredictor_2 = _mod.FixationsPredictor_2

from typing import TypeVar
T = TypeVar("T", bound="Module")
import re

try:
    from mixture_module import MixtureTokenModule
    _MIXTURE_AVAILABLE = True
except ImportError:
    _MIXTURE_AVAILABLE = False
    MixtureTokenModule = None

class MyRewardBase:
    def __init__(
        self,
        model_name,
        features_used=[1, 1, 1, 1, 1],
        *argv,
        **karg,
    ):
        self.features_used = features_used
        self.model_name = model_name
        self.use_mixture_token = False
        self.mixture_module = None

        self.memory_storage = LMDBStorage(
            db_path=os.environ.get(
                "LMDB_CACHE_PATH",
                os.path.join(os.getcwd(), "buffer_train.lmdb"),
            )
        )

    def enable_mixture_token(
        self,
        hidden_size,
        K=3,
        cov_type="diag",
        proj_hidden=128,
        dropout=0.1,
        log_transform=True,
    ):
        if not _MIXTURE_AVAILABLE:
            raise ImportError("mixture_module not available")
        num_features = int(sum(self.features_used))
        self.mixture_module = MixtureTokenModule(
            num_features=num_features,
            K=K,
            hidden_size=hidden_size,
            cov_type=cov_type,
            proj_hidden=proj_hidden,
            dropout=dropout,
            log_transform=log_transform,
        )
        self.use_mixture_token = True
        self.mixture_K = K
        self.mixture_cov_type = cov_type

    def _load_tokenizer(self, load_local_folder_name=None):
        if load_local_folder_name:
            tokenizer = AutoTokenizer.from_pretrained(load_local_folder_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
        if tokenizer.chat_template is None:
            tokenizer.chat_template = tokenizer.default_chat_template
        tokenizer.add_eos_token = True
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        tokenizer.padding_side = "right"
        chat_tokens = list(set(re.findall(r"(<.*?>)", tokenizer.default_chat_template)))
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": tokenizer.additional_special_tokens
                + chat_tokens
            }
        )
        self.tokenizer = tokenizer
        return self.tokenizer

    def load_fx_model_1(self, hidden_size, remap=False, fp_dropout=[0.0, 0.3]):
        p_1, p_2 = fp_dropout
        self.modelTokenizer = self.tokenizer
        self.FP_model = FixationsPredictor_1(
            hidden_dim=128,
            drop_out=0.2,
            modelTokenizer=self.modelTokenizer,
            remap=remap,
        )
        self.fixations_embedding_projector = nn.Sequential(
            nn.Linear(1, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(p=p_1),
            nn.Linear(128, hidden_size),
            nn.Dropout(p=p_2),
        )
        self.norm_layer_fix = nn.LayerNorm(hidden_size)

    def load_fx_model_2(
        self,
        hidden_size,
        remap=False,
        fp_dropout=[0.0, 0.3],
        load_fix_model=True,
    ):
        p_1, p_2 = fp_dropout
        self.modelTokenizer = self.tokenizer
        if load_fix_model:
            self.FP_model = FixationsPredictor_2(
                modelTokenizer=self.modelTokenizer, remap=remap
            )
        num_features = int(sum(self.features_used))
        self.fixations_embedding_projector = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(p=p_1),
            nn.Linear(128, hidden_size),
            nn.Dropout(p=p_2),
        )
        self.norm_layer_fix = nn.LayerNorm(hidden_size)

    def _compute_fixations(
        self, input_ids, attention_mask, remap=False, fixations_model_version=1
    ):
        if fixations_model_version == 1:
            (
                fixations,
                fixations_attention_mask,
                mapped_fixations,
                text_tokenized_model,
                text_tokenized_fix,
                sentences,
            ) = self.FP_model._compute_mapped_fixations(input_ids)
        elif fixations_model_version == 2:
            (
                fixations,
                fixations_attention_mask,
                mapped_fixations,
                text_tokenized_model,
                text_tokenized_fix,
                sentences,
            ) = self.FP_model._compute_mapped_fixations(input_ids, attention_mask)
        if remap:
            fixations_attention_mask = attention_mask
        return (
            fixations,
            fixations_attention_mask,
            mapped_fixations,
            text_tokenized_model,
            text_tokenized_fix,
            sentences,
        )

    def compute_fixations(
        self, input_ids, attention_mask, remap=False, fixations_model_version=1
    ):
        (
            fixations,
            fixations_attention_mask,
            mapped_fixations,
            text_tokenized_model,
            text_tokenized_fix,
            sentences,
        ) = self.compute_fixations_cached(
            input_ids, attention_mask, remap, fixations_model_version
        )
        del text_tokenized_fix, text_tokenized_model, sentences

        mixture_token = None
        mixture_mask = None
        if self.use_mixture_token and self.mixture_module is not None:
            mix_in = fixations
            if fixations_model_version == 1 and mix_in.dim() == 2:
                mix_in = mix_in.unsqueeze(-1)
            mixture_token, mixture_mask = self.mixture_module(
                mix_in, fixations_attention_mask
            )

        fixations_normalized, fixations_attention_mask = self.process_fixations(
            fixations,
            fixations_attention_mask,
            mapped_fixations,
            remap,
            fixations_model_version,
        )

        if self.use_mixture_token:
            return (
                fixations_normalized,
                fixations_attention_mask,
                mixture_token,
                mixture_mask,
            )
        return fixations_normalized, fixations_attention_mask

    def process_fixations(
        self,
        fixations,
        fixations_attention_mask,
        mapped_fixations,
        remap,
        fixations_model_version,
    ):
        if remap:
            fixations = mapped_fixations
            del mapped_fixations
        if self.training is False and self.noise_factor > 0:
            noise = torch.randn_like(fixations) * self.noise_factor
            fixations = fixations + noise
            noise = noise.detach()
            del noise
        if fixations_model_version == 1:
            fixations = fixations.unsqueeze(2)
        fixations_projected = self.fixations_embedding_projector(fixations)
        fixations_normalized = self.norm_layer_fix(fixations_projected)
        torch.cuda.empty_cache()
        return fixations_normalized, fixations_attention_mask

    @staticmethod
    def hash_value(val):
        return hashlib.md5(str(val).encode()).hexdigest()

    @staticmethod
    def remove_padding_from_batch(batch_token_ids, pad_token_id=0):
        return [
            list(filter(lambda token_id: token_id != pad_token_id, sequence))
            for sequence in batch_token_ids
        ]

    def compute_fixations_cached(
        self, input_ids_original, attention_mask, remap=False, fixations_model_version=1
    ):
        device = input_ids_original.device
        input_ids_list = input_ids_original.cpu().numpy().tolist()
        filtered_ids = self.remove_padding_from_batch(
            input_ids_list, self.tokenizer.pad_token_id
        )
        fixations_all, fixations_attention_mask_all = [], []
        for seq in filtered_ids:
            if remap is True:
                hash_id = self.hash_value(
                    seq + [fixations_model_version] + ["remap"] + [self.model_name]
                )
            else:
                hash_id = self.hash_value(
                    seq + [fixations_model_version] + [self.model_name]
                )

            result = self.memory_storage.getItem(hash_id)

            if result is None:
                torch_seq = torch.LongTensor(np.asarray(seq)).to(device).unsqueeze(0)
                (
                    fixations,
                    fixations_attention_mask,
                    mapped_fixations,
                    text_tokenized_model,
                    text_tokenized_fix,
                    sentences,
                ) = self._compute_fixations(
                    torch_seq,
                    attention_mask,
                    remap=remap,
                    fixations_model_version=fixations_model_version,
                )
                del text_tokenized_fix, text_tokenized_model, sentences
                if remap:
                    fixations = mapped_fixations
                    fixations_attention_mask = attention_mask
                fixation_outputs = {
                    "fixations": fixations.cpu(),
                    "fixations_attention_mask": fixations_attention_mask.cpu(),
                }
                self.memory_storage.add(hash_id, fixation_outputs)
            else:
                fixations = result["fixations"].to(device)
                fixations_attention_mask = result["fixations_attention_mask"].to(device)
            if fixations_model_version == 2:
                idx = np.where(np.array(self.features_used) == 1)[0]
                fixations = fixations[:, :, idx]
            fixations_all.append(fixations.squeeze())
            fixations_attention_mask_all.append(fixations_attention_mask.squeeze())

        fixations_all = self._pad_and_concat(fixations_all)
        if remap is False:
            fixations_attention_mask_all = self._pad_and_concat(
                fixations_attention_mask_all
            )
            return fixations_all, fixations_attention_mask_all, None, None, None, None
        else:
            try:
                fixations_attention_mask_all = self._pad_and_concat(
                    fixations_attention_mask_all
                )
            except Exception:
                pass
            fixations_attention_mask_all = attention_mask
            return None, fixations_attention_mask_all, fixations_all, None, None, None

    @staticmethod
    def _pad_and_concat(list_of_tensors):
        def pad_tensor(tensor, max_length):
            padding_length = max_length - tensor.size(0)
            if padding_length > 0:
                if tensor.dim() == 1:
                    padding_tensor = torch.zeros(padding_length).to(tensor.device)
                elif tensor.dim() == 2:
                    padding_tensor = torch.zeros(padding_length, tensor.size(1)).to(
                        tensor.device
                    )
                else:
                    raise ValueError("Only 1D and 2D tensors are supported.")
                tensor = torch.cat([tensor, padding_tensor])
            return tensor

        max_length = max([len(i) for i in list_of_tensors])
        return torch.stack([pad_tensor(item, max_length) for item in list_of_tensors])
