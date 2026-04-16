import os
import sys
import threading
from datetime import datetime

# sys.path.append('/home/csp/repo/LLMs/eye_transformer/')
# print("CWD", os.getcwd(), "PATH", sys.path)

import pathlib

sys.path.append("../..")

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
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from transformers import (
    LlamaForSequenceClassification,
    AutoModelForSequenceClassification,
)
from transformers.modeling_outputs import (
    SequenceClassifierOutputWithPast,
)

from typing import (
    TypeVar,
)
from reward_model_base import MyRewardBase

T = TypeVar("T", bound="Module")
import re

def create_dynamic_class_RewardConcatenate(base_class=LlamaForSequenceClassification):
    class MyRewardConcatenate(base_class, MyRewardBase):
        def __init__(
            self,
            model_name,
            bnb_config=False,
            use_softprompt=True,
            load_local_folder_name=None,
            fixations_model_version=1,
            noise_factor=0.0,
            fp_dropout=[0.0, 0.3],
            load_fix_model=True,
            features_used=[1, 1, 1, 1, 1],

            use_mixture_token=False,
            mixture_K=3,
            mixture_cov_type="diag",
            mixture_proj_hidden=128,
            mixture_dropout=0.1,
            mixture_log_transform=True,
            *argv,
            **karg,
        ):
            print("loading model", model_name)
            start = datetime.now()
            use_quantization = bnb_config not in (None, False)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=1,
                quantization_config=bnb_config if use_quantization else None,
                device_map="auto",
                *argv,
                **karg,
            )
            config = model.config
            super(MyRewardConcatenate, self).__init__(
                config,
                *argv,
                **karg,
            )
            MyRewardBase.__init__(
                self, model_name=model_name, features_used=features_used
            )
            end = datetime.now()
            print("Total time loading model", end.timestamp() - start.timestamp())
            self.model_name = model_name
            self.model = model.model
            self.noise_factor = noise_factor
            self.fp_dropout = fp_dropout
            self.use_softprompt = use_softprompt
            self.fixations_model_version = fixations_model_version
            self.use_quantization = use_quantization
            self.bnb_config = bnb_config
            self._load_tokenizer(load_local_folder_name)
            self.thread_local = threading.local()
            self.load_fix_model = load_fix_model

            if self.use_softprompt:
                if self.fixations_model_version == 1:
                    self.load_fx_model_1(
                        config.hidden_size, fp_dropout=self.fp_dropout, remap=False
                    )
                elif self.fixations_model_version == 2:
                    self.load_fx_model_2(
                        config.hidden_size,
                        fp_dropout=self.fp_dropout,
                        remap=False,
                        load_fix_model=self.load_fix_model,
                    )
                else:
                    raise ValueError(
                        f"Fixations model version {self.fixations_model_version} not supported"
                    )
                self.concat_tokens = ["<eye/>", "</eye>"]

                self.tokenizer.add_tokens(self.concat_tokens)
                self.concat_tokens_ids = [
                    self.tokenizer.convert_tokens_to_ids(x) for x in self.concat_tokens
                ]

            if use_mixture_token:
                self.enable_mixture_token(
                    hidden_size=config.hidden_size,
                    K=mixture_K,
                    cov_type=mixture_cov_type,
                    proj_hidden=mixture_proj_hidden,
                    dropout=mixture_dropout,
                    log_transform=mixture_log_transform,
                )
            self.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.resize_token_embeddings(len(self.tokenizer))

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
            r"""
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
            """
            inputs_embeds = self.model.embed_tokens(input_ids.to("cuda"))
            attention_mask = attention_mask.to("cuda")
            if self.use_softprompt:

                if self.use_mixture_token:
                    (
                        fixations_normalized,
                        fixations_attention,
                        mixture_token,
                        mixture_mask,
                    ) = self.compute_fixations(
                        input_ids,
                        attention_mask,
                        remap=False,
                        fixations_model_version=self.fixations_model_version,
                    )
                else:
                    fixations_normalized, fixations_attention = self.compute_fixations(
                        input_ids,
                        attention_mask,
                        remap=False,
                        fixations_model_version=self.fixations_model_version,
                    )
                    mixture_token = None
                    mixture_mask = None

                concat_tokens_embed = self.model.embed_tokens(
                    torch.tensor(self.concat_tokens_ids).to("cuda")
                )
                concat_tokens_embed_start = (
                    concat_tokens_embed[0]
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .expand(fixations_normalized.shape[0], -1, -1)
                )
                concat_tokens_embed_end = (
                    concat_tokens_embed[1]
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .expand(fixations_normalized.shape[0], -1, -1)
                )
                separator_attention_mask = (
                    torch.tensor([1])
                    .expand(fixations_normalized.shape[0], -1)
                    .to("cuda")
                )

                embed_chunks = []
                mask_chunks = []
                if mixture_token is not None:
                    embed_chunks.append(mixture_token.to(inputs_embeds.dtype).to("cuda"))
                    mask_chunks.append(mixture_mask.to("cuda"))
                embed_chunks.extend([
                    concat_tokens_embed_start,
                    fixations_normalized,
                    concat_tokens_embed_end,
                    inputs_embeds,
                ])
                mask_chunks.extend([
                    separator_attention_mask,
                    fixations_attention,
                    separator_attention_mask,
                    attention_mask,
                ])
                inputs_embeds = torch.cat(embed_chunks, dim=1)
                attention_mask = torch.cat(mask_chunks, dim=1)

                del (
                    concat_tokens_embed_start,
                    concat_tokens_embed_end,
                    fixations_normalized,
                    separator_attention_mask,
                    fixations_attention,
                )
            else:
                inputs_embeds = inputs_embeds.float()

            output = super().forward(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            torch.cuda.empty_cache()
            return output

    return MyRewardConcatenate
