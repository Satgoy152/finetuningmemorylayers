# qwen_mem/qwen_with_memory.py
import torch
from torch import nn

from transformers import AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from memory_layer import MemoryPlus


class DecoderWithMemory(nn.Module):
    """
    Wraps an existing Qwen2DecoderLayer and adds a shared MemoryPlus
    after the normal layer forward.
    """

    def __init__(self, base_layer: Qwen2DecoderLayer, memory: MemoryPlus):
        super().__init__()
        self.base_layer = base_layer
        self.memory = memory

        # keep attributes used by HF utils
        self.attention_type = base_layer.attention_type

    def forward(self, hidden_states, *args, **kwargs):
        # Run the original decoder layer
        hidden_states = self.base_layer(hidden_states, *args, **kwargs)
        # Add memory output (residual-style)
        hidden_states = hidden_states + self.memory(hidden_states)
        mem_out = self.memory(hidden_states)
        if not hasattr(self, "_printed"):
            print("Memory output norm:", mem_out.norm().item())
            self._printed = True
        return hidden_states


def load_qwen_with_memory(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    layers_with_memory = (4, 10, 16),
    n_mem: int = 10_000,
    k: int = 32,
):
    """
    Load Qwen2.5-0.5B and wrap a few decoder layers with a shared MemoryPlus.

    Returns:
        model:     patched AutoModelForCausalLM
        memory:    shared MemoryPlus module
    """
    # Load the usual Qwen model from HF
    model = AutoModelForCausalLM.from_pretrained(model_name)
    d_model = model.config.hidden_size

    # Create ONE shared MemoryPlus
    memory = MemoryPlus(
        d_model=d_model,
        n_mem=n_mem,
        d_key=256,
        d_value=d_model,
        k=k,
    )

    # Replace selected decoder layers with wrappers
    for idx, layer in enumerate(model.model.layers):
        if idx in layers_with_memory:
            model.model.layers[idx] = DecoderWithMemory(layer, memory)

    return model, memory


if __name__ == "__main__":
    model, memory = load_qwen_with_memory()
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # dummy forward
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    inputs = tok("hello world", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    print("OK, forward pass with memory worked. logits shape:", out.logits.shape)
