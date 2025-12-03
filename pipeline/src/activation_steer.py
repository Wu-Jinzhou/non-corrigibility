import torch
from contextlib import contextmanager
from typing import Iterable, Sequence, Union


class ActivationSteerer:
    """
    Add (coeff * steering_vector) to a chosen transformer block's output.
    Minimal dependency copy from persona_vectors to keep demo self-contained.
    """

    _POSSIBLE_LAYER_ATTRS: Iterable[str] = (
        "transformer.h",       # GPT‑2/Neo, Bloom, etc.
        "encoder.layer",       # BERT/RoBERTa
        "model.layers",        # Llama/Mistral
        "gpt_neox.layers",     # GPT‑NeoX
        "block",               # Flan‑T5
    )

    def __init__(
        self,
        model: torch.nn.Module,
        steering_vector: Union[torch.Tensor, Sequence[float]],
        *,
        coeff: float = 1.0,
        layer_idx: int = -1,
        positions: str = "all",
        debug: bool = False,
    ):
        self.model, self.coeff, self.layer_idx = model, float(coeff), layer_idx
        self.positions = positions.lower()
        self.debug = debug
        self._handle = None

        p = next(model.parameters())
        self.vector = torch.as_tensor(steering_vector, dtype=p.dtype, device=p.device)
        if self.vector.ndim != 1:
            raise ValueError("steering_vector must be 1-D")
        hidden = getattr(model.config, "hidden_size", None)
        if hidden and self.vector.numel() != hidden:
            raise ValueError(f"Vector length {self.vector.numel()} ≠ model hidden_size {hidden}")

        valid_positions = {"all", "prompt", "response"}
        if self.positions not in valid_positions:
            raise ValueError("positions must be 'all', 'prompt', 'response'")

    def _locate_layer(self):
        for path in self._POSSIBLE_LAYER_ATTRS:
            cur = self.model
            for part in path.split("."):
                if hasattr(cur, part):
                    cur = getattr(cur, part)
                else:
                    break
            else:
                if not hasattr(cur, "__getitem__"):
                    continue
                if not (-len(cur) <= self.layer_idx < len(cur)):
                    raise IndexError("layer_idx out of range")
                if self.debug:
                    print(f"[ActivationSteerer] hooking {path}[{self.layer_idx}]")
                return cur[self.layer_idx]
        raise ValueError("Could not find layer list on the model.")

    def _hook_fn(self, module, ins, out):
        steer = self.coeff * self.vector

        def _add(t):
            if self.positions == "all":
                return t + steer.to(t.device)
            elif self.positions == "prompt":
                return t + steer.to(t.device)
            elif self.positions == "response":
                t2 = t.clone()
                t2[:, -1, :] += steer.to(t.device)
                return t2
            else:
                return t

        if torch.is_tensor(out):
            new_out = _add(out)
        elif isinstance(out, (tuple, list)):
            if not torch.is_tensor(out[0]):
                return out
            head = _add(out[0])
            new_out = (head, *out[1:])
        else:
            return out
        return new_out

    def __enter__(self):
        layer = self._locate_layer()
        self._handle = layer.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, *exc):
        self.remove()

    def remove(self):
        if self._handle:
            self._handle.remove()
            self._handle = None


@contextmanager
def multi_steerer(model, vectors, layers, coeff=1.0, positions="all"):
    """
    Apply the same coeff/positions across a list of layers.
    vectors: tensor [layers+1, hidden]; layers: list of layer indices.
    """
    steerers = []
    try:
        for layer in layers:
            v = vectors[layer]
            steerers.append(
                ActivationSteerer(
                    model,
                    steering_vector=v,
                    coeff=coeff,
                    layer_idx=layer - 1,  # steer block index (vector index includes embedding)
                    positions=positions,
                )
            )
        for s in steerers:
            s.__enter__()
        yield
    finally:
        for s in steerers[::-1]:
            s.__exit__(None, None, None)
