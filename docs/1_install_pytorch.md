# Install PyTorch

## Conda update Python to 3.6
- https://www.scivision.co/switch-anaconda-python-36/

```bash
Last login: Tue Aug 21 09:55:52 on console
pwd
% pwd
/Users/reshamashaikh
% python --version
Python 3.5.2 :: Anaconda custom (x86_64)
```
```bash
% conda install python=3.6
Fetching package metadata ...........
```

## PyTorch Install
- https://pytorch.org
- OS:  MacOS
- Package Manager:  `conda`
- Python:  3.6
- CUDA:  ? None
- Run this command:
```bash
conda install pytorch torchvision -c pytorch
```
- output
```bash
% pytorch --version       
zsh: command not found: pytorch
% conda install pytorch torchvision -c pytorch
```
- post-install
```bash
% python --version
Python 3.6.5 :: Anaconda custom (64-bit)
```

## Check PyTorch Install
```bash
% python3
Python 3.6.5 |Anaconda custom (64-bit)| (default, Apr 26 2018, 08:42:37) 
[GCC 4.2.1 Compatible Clang 4.0.1 (tags/RELEASE_401/final)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> print(torch.__version__)
0.4.1
>>> exit()
%
```
  
### CUDA Info
- https://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html

