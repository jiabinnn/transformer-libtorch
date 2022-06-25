# transformer-libtorch

a simple example for torch transformer english to french translate.
try implement inference in linux c++ environment.

## env

ubuntu18.04
python == 3.9.12
torch == 1.11.0
cuda == 11.3
cudnn == cudnn-11.3-linux-x64-v8.2.1.32
libtorch == libtorch-cxx11-abi-shared-with-deps-1.11.0+cu113 or libtorch-cxx11-abi-shared-with-deps-1.11.0+cpu (change device=cpu in config/config.ini)

## run

`cd path_to_your_workspace`
change configs in config/config.ini if you want

### train

`python transformer/trainer.py` will save the final model file in saved_models/

### inference

1. run `python transformer/save_model_trace.py`, you will get a trace model file in saved_models/
2. compile c++ program
   - you should firstly change your libtorch path in CMakeLists.txt, `set(CMAKE_PREFIX_PATH "/home/tars/libtorch/libtorch_gpu")`.
   - `mkdir build && cd build && cmake .. && make`, you get the bin file in path_to_your_workspace/bin
   - `cd/path_to_your_workspace` and run `./bin/main`,input english sentence, press enter to inference, press q to quit.
