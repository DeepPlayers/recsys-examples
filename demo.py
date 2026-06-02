import os
import torch

path = os.path.abspath("corelib/dynamicemb/torch_binding_build/inference_emb_ops.so")
torch.ops.load_library(path)
print("loaded", path)
