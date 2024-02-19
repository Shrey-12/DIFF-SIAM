import torch

use_cuda = torch.cuda.is_available()

def gpu_test():
	from torch import Tensor
	x: Tensor = torch.randn(2, 4).cuda()
	y: Tensor = torch.randn(4, 1).cuda()
	out: Tensor = (x @ y)
	assert out.size() == torch.Size([2, 1])
	print(f'Success, no Cuda errors means it worked see:\n{out=}')

if use_cuda:
	print('__CUDNN VERSION:', torch.backends.cudnn.version())
	print('__Number CUDA Devices:', torch.cuda.device_count())
	print('__CUDA Device Name:',torch.cuda.get_device_name(0))
	print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)
	gpu_test()
