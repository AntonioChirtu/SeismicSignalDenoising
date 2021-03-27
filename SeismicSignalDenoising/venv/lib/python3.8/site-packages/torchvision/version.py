__version__ = '0.9.0'
git_version = '01dfa8ea81972bb74b52dc01e6a1b43b26b62020'
from torchvision.extension import _check_cuda_version
if _check_cuda_version() > 0:
    cuda = _check_cuda_version()
