import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Set proxy settings
os.environ['HTTP_PROXY'] = 'http://proxy-dmz.intel.com:912/'
os.environ['HTTPS_PROXY'] = 'http://proxy-dmz.intel.com:912/'

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").to(torch_device)

model_inputs = tokenizer("An explanation of Linear Regression", return_tensors="pt").to(torch_device)
output = model.generate(**model_inputs, max_new_tokens=50, do_sample=True, top_p=0.92, top_k=0, temperature=0.6)

print(tokenizer.decode(output[0], skip_special_tokens=True))