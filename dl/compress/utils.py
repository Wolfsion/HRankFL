import re
import torch


# Calculate memory size of the give tensor - units:B
def get_size(tensor: torch.Tensor) -> int:
    nums_element = tensor.nelement()
    regex = r"(\d+)$"
    test_str = str(tensor.dtype)
    match = next(re.finditer(regex, test_str, re.MULTILINE))
    bits = int(match.group())
    return bits * nums_element // 8


