import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_path = "/home/zhsu/jyluo/models/gemma-7b"


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    model_path
)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("Model loaded successfully!")

print("Device:")
print(model.device)

print("GPU memory allocated:")
print(
    torch.cuda.memory_allocated() / 1024**3,
    "GB"
)