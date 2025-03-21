import os
from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import logging
import torch

# Set proxy settings
os.environ['HTTP_PROXY'] = 'http://proxy-dmz.intel.com:912/'
os.environ['HTTPS_PROXY'] = 'http://proxy-dmz.intel.com:912/'

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Configure logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", 
    level=logging.INFO, 
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Define model directories
pretrained_model_dir = "facebook/opt-125m"
quantized_model_dir = "opt-125m-4bit"

# Download the tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)

# Example inputs for quantization
examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm.",
        return_tensors="pt"
    ).to(torch_device)
]

# Create quantization configuration
quantize_config = BaseQuantizeConfig(
    bits=4,  # Quantize model to 4-bit
    group_size=128,  # Recommended to set the value to 128
    desc_act=False,  # Set to False can significantly speed up inference but the perplexity may slightly worsen
)

# Load un-quantized model (ensure CUDA is available)
if torch_device == 'cuda':
    model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config).to(torch_device)
else:
    raise EnvironmentError("CUDA is not available. Quantization requires CUDA.")

# Quantize model (examples should be a list of dicts with keys "input_ids" and "attention_mask")
model.quantize(examples)

# Save quantized model
model.save_quantized(quantized_model_dir)

# Save quantized model using safetensors
model.save_quantized(quantized_model_dir, use_safetensors=True)

# Load the quantized model to the first GPU
model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0")

# Example usage with model.generate
input_text = tokenizer("auto_gptq is", return_tensors="pt").to(model.device)
output = model.generate(**input_text)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# Alternatively, use the pipeline
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
print(pipeline("auto-gptq is")[0]["generated_text"])

