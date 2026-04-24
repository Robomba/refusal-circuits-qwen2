import torch
from transformer_lens import HookedTransformer

# Free the base model first
del model
torch.cuda.empty_cache()

print("Loading Qwen2-1.5B-Instruct...")
model = HookedTransformer.from_pretrained(
    "Qwen/Qwen2-1.5B-Instruct",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    dtype="bfloat16",
    device="cuda"
)
print(f"Instruct model loaded!")
print(f"GPU memory used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
