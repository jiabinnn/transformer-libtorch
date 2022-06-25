import torch
from config import MyConfig

config = MyConfig('config/config.ini')

model_entire_path = config.model_entire_path
model_trace_path = config.model_trace_path
max_len = config.max_len
device = config.device

model = torch.load(model_entire_path).to(device)
model.eval()

demo_encoder_inputs = torch.zeros((1, max_len), dtype=torch.long).to(device)
demo_decoder_inputs = torch.zeros((1, max_len), dtype=torch.long).to(device)
mask = torch.ones((1, max_len), dtype=torch.bool).to(device)
traced_script_module = torch.jit.trace(model, (demo_encoder_inputs, demo_decoder_inputs, mask, mask, mask))
traced_script_module.save(model_trace_path)

# model_poetry_stat_dict_path = config.model_poetry_stat_dict_path
# model_poetry_entire_path = config.model_poetry_entire_path
# model_poetry_trace_path = config.model_poetry_trace_path

# max_len = config.max_len

# model = torch.load(model_poetry_entire_path)
# model.eval()

# demo_encoder_inputs = torch.zeros((1, max_len), dtype=torch.long)
# demo_decoder_inputs = torch.zeros((1, max_len), dtype=torch.long)
# mask = torch.ones((1, max_len), dtype=torch.bool)
# traced_script_module = torch.jit.trace(model, (demo_encoder_inputs, demo_decoder_inputs, mask, mask, mask))
# traced_script_module.save(model_poetry_trace_path)
