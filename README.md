# Model Merger Script

This Python script is designed to merge the weights from multiple model files into a single model file. It supports a variety of machine learning model formats, including PyTorch, TensorFlow, ONNX, CoreML, and others. Additionally, it provides the option to use memory-mapped files for faster loading of large models in certain formats.

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/sp4s-s/mlmodel-merger.git
cd model_merger
```
### Step 2: Install Dependencies
Install the required packages listed in requirements.txt:
```bash
pip install -r requirements.txt
```
This will install all the necessary Python libraries, including PyTorch, TensorFlow, Safetensors, ONNX, CoreML, and others.

Requirements
The dependencies for running model_merger.py are listed in requirements.txt:

-------------------------------------------------------------------------------------------------------
These packages cover:

- PyTorch for .pt and .jit.pt models.
- TensorFlow for .pb, .savedmodel, and .h5 models.
- Safetensors for handling .safetensors models.
- ONNX for .onnx models.
- CoreMLTools for .mlmodel.
- OpenVINO for .xml models.
- argparse for command-line argument parsing.
- protobuf and h5py for handling models and weights.


Usage
Run the script from the command line as follows:

```bash
$ python model_merger.py [-h] [-i INPUT_DIR] [-o OUTPUT_FILE] [-t MODEL_TYPE] [-m]

Command-line Arguments
    -h, --help: Show the help message and exit.
    -i, --input_dir: Required: Specify the directory containing model files. If this argument is not given, the current directory will be used by default.
    -o, --output_file: Required: Provide the name for the output file (without extension).
    -t, --model_type: Optional: Specify the model file type/extension to be processed. The script will merge only files with the specified extension. Supported extensions include:
    .pb (TensorFlow)
    .pt, .jit.pt (PyTorch)
    .onnx (ONNX)
    .h5 (TensorFlow/Keras)
    .prototxt, .param (Caffe)
    .savedmodel (TensorFlow)
    .mlmodel (CoreML)
    .plan (TensorRT)
    .xml (OpenVINO)
    .safetensors (Safetensors)
```
If this argument is not provided, the script will process all supported model files in the directory.
-m, --mmap: Optional: Enable memory-mapped files for .pth and .pt model files. This helps optimize memory usage when dealing with large models.
Example Usage
Merge weights from all supported models in a directory:
```bash

python model_merger.py -i /path/to/models -o merged_model
```
Merge only PyTorch models:
```bash

python model_merger.py -i /path/to/models -o merged_model -t .pt
```
Use memory-mapped files for large PyTorch models:
```bash

python model_merger.py -i /path/to/models -o merged_model -t .pt -m
```
Merge models across all types (no file type specified):
```bash

python model_merger.py -i /path/to/models -o merged_model
```
Supported Model Formats
The script supports the following model file formats for merging:

PyTorch: .pt, .jit.pt, .pth
TensorFlow: .pb, .savedmodel, .h5
ONNX: .onnx
CoreML: .mlmodel
Caffe: .prototxt, .param
TensorRT: .plan
OpenVINO: .xml
Safetensors: .safetensors

[][][][] Key Notes </br>
> Model Merging: The script combines the weights of models by summing the weights for matching keys. This assumes that the models have the same architecture and compatible layers.
> Memory Mapping (-m): The -m argument should be used for .pt and .pth PyTorch models to reduce memory usage when loading large models.
> Saving Models: The script saves the merged model in the same format as the input model type. For some formats (e.g., .pb, .onnx, .mlmodel), you may need additional libraries or conversion tools.

License
This project is licensed under the PRIVATE License - see the LICENSE file for details.
