"""
This file is part of midas-cpp

MIT License

Copyright (c) 2020 Shing Yan Loo (lsyan@ualberta.ca)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sys
import os

import torch

def convert(in_model_path, out_model_path):
    """Run MiDaS to create the torchscript model.
    https://pytorch.org/tutorials/advanced/cpp_export.html

    :param in_model_path: the path to the pre-trained model
    :param out_model_path: the output filename
    """
    # select device
    device = torch.device("cuda")
    print("device: %s" % device)

    # load network
    example_input = torch.rand(1, 3, 288, 384, dtype=torch.float32)
    model = MidasNet(in_model_path, non_negative=True)
    sm = torch.jit.trace(model, example_input)

    # Save the torchscript model
    sm.save(out_model_path)

def is_valid_extension(path):
    """
    Make sure that the provided path has the correct extension
    """
    if os.path.split(path)[1].endswith('pt'):
        return True
    else:
        return False

if __name__ == "__main__":
    # Check if the number of arguments is correct
    if len(sys.argv) != 4:
        print(len(sys.argv))
        print('Usage: python3 export_model.py <MiDaS root folder> <pre-trained model file from MiDaS> <output filename>')
        exit(0)

    # Get the filenames
    in_model_path = sys.argv[2]
    out_model_path = sys.argv[3]
    if not is_valid_extension(in_model_path):
        print('Incorrect pretrained model file extension. Has to be a .pt file.')
        exit(0)
    if not is_valid_extension(out_model_path):
        print('Incorrect output model file extension. Has to be a .pt file.')
        exit(0)
    print("Pre-trained model path:", in_model_path)
    print("Output model path:", out_model_path)
    
    # Load the MiDaS module
    try:
        print('MiDaS root folder:', sys.argv[1])
        sys.path.insert(1, sys.argv[1])
        from midas.midas_net import MidasNet
        print('Successfully loaded MiDasNet.')
    except ImportError:
        print('Failed to import MiDasNet. Please check the path to the root folder of MiDaS.')
        exit(0)
    
    # Convert the model to a torchscript model
    convert(in_model_path, out_model_path)
