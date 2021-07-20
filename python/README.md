# Export the pre-trained MiDaS model to a torchscript model
Before we can run the midascpp Boost.Python module or MiDaS-cpp, we need to convert the model to a torchscript model. To perform the conversion, we need to run the `export_model.py` script:

```shell
python3 export_model.py <MiDaS root folder> <pre-trained model file from MiDaS> <output filename>
```

For example:

```shell
python3 export_model.py $HOME/MiDaS $HOME/MiDaS/model-f45da743.pt $HOME/MiDaS-cpp/traced_model.pt
```
***Make sure you download the [MiDaS](https://github.com/intel-isl/MiDaS) repo as well as their pre-trained model before running this script. If the `midas.so` file is present in this folder, move the file to a temporary folder as it will cause a problem due to module name ambiguity.***

# (Optional) Test the midas Boost.Python module
At this point, you should have built the project and generated a `midas.so` file in this folder. To perform depth prediction, run `python3 inference.py` 

