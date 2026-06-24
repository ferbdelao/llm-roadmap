"""
Fragmento ilustrativo:
  1) Cómo se ve, conceptualmente, una capa lineal con un adaptador LoRA.
  2) Cómo se configuraría un fine-tuning con LoRA usando la librería
     `peft` de Hugging Face sobre un modelo real.

Requisitos sugeridos para la parte 2:
    pip install transformers peft accelerate torch
"""

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Capa lineal congelada + adaptador LoRA de bajo rango.

    W' = W + (alpha / r) * B @ A

    W permanece congelada (requires_grad=False); solo A y B se entrenan.
    """

    def __init__(self, in_features: int, out_features: int, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base_linear = nn.Linear(in_features, out_features, bias=False)
        self.base_linear.weight.requires_grad = False  # <- pesos originales congelados

        # Adaptadores de rango bajo: A reduce la dimensión, B la regresa al tamaño original
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, r))  # B inicia en cero (sin efecto al inicio)

        self.scaling = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        lora_out = (x @ self.A.T) @ self.B.T  # equivalente a x @ (B @ A)^T
        return base_out + self.scaling * lora_out


if __name__ == "__main__":
    layer = LoRALinear(in_features=64, out_features=64, r=4, alpha=8)

    trainable_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in layer.parameters())
    print(f"Parámetros entrenables: {trainable_params} / {total_params} "
          f"({100 * trainable_params / total_params:.1f}%)")

    x = torch.randn(2, 64)
    out = layer(x)
    print("Forma de salida:", out.shape)


# --------------------------------------------------------------------------
# Ejemplo de configuración real con la librería `peft` (Hugging Face)
# para afinar un modelo preentrenado completo. Descomenta para usarlo.
# --------------------------------------------------------------------------
#
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import LoraConfig, get_peft_model
#
# model_name = "mistralai/Mistral-7B-v0.1"
# model = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
#
# lora_config = LoraConfig(
#     r=8,                              # rango de las matrices A y B
#     lora_alpha=16,                    # factor de escala
#     target_modules=["q_proj", "v_proj"],  # a qué capas se le añade LoRA
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
# )
#
# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()
# # Salida típica: "trainable params: 4,194,304 || all params: 7,245,000,000 || trainable%: 0.058"
