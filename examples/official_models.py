"""Adapters only: instantiate upstream implementations without replacing forward()."""
import torch
from torch import nn
from upstream.set_transformer.modules import SAB, ISAB
from pathlib import Path
import json

def provenance(model):
    model._silicondevine_source=json.loads((Path(__file__).parent/'upstream/set_transformer/SOURCE.json').read_text())
    return model


def build_sab():
    torch.manual_seed(73)
    return {'model':provenance(SAB(16,16,4,ln=True)),'args':(torch.randn(1,6,16),),'options':{'backend':'export'}}


def build_isab():
    torch.manual_seed(73)
    return {'model':provenance(ISAB(16,16,4,3,ln=True)),'args':(torch.randn(1,6,16),),'options':{'backend':'export'}}


def build_llama():
    from transformers import LlamaConfig,LlamaForCausalLM
    torch.manual_seed(73)
    config=LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=32,use_cache=False,attn_implementation='eager')
    model=LlamaForCausalLM(config).eval()
    import transformers
    model._silicondevine_source={'repository':'https://github.com/huggingface/transformers','version':transformers.__version__,'implementation':'transformers.models.llama.modeling_llama.LlamaForCausalLM','weights':'random initialization; reduced config; no pretrained weights'}
    tokens=torch.tensor([[1,7,3,12,9,4]])
    # Explicit model inputs avoid tracing Python mask-construction control flow.
    mask=torch.full((1,1,6,6),float('-inf')).triu(1)
    return {'model':model,'args':(tokens,),'options':{'backend':'export','kwargs':{'attention_mask':mask,'use_cache':False,'return_dict':False}}}

build_model=build_llama
