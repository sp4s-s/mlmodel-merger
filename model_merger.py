# python model_merger.py [-h] [-i INPUT_DIR] [-o OUTPUT_FILE] [-t MODEL_TYPE (if this is empty than use brute force do operation for all types ml file found)] [-m {for using mmap_merge_weights() if not given -m than merge_weights()}]

import os
import sys
import time
import argparse
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
import mmap

SUPPORTED_EXTENSIONS = [
    ".pb", 
    ".pt", 
    ".onnx", 
    ".h5", 
    ".prototxt", 
    ".param", 
    ".savedmodel", 
    ".jit.pt", 
    ".mlmodel", 
    ".plan", 
    ".xml", 
    ".safetensors",
]

def get_file_list(dir_path, file_ext=None):
    files = []
    for root, _, filenames in os.walk(dir_path):
        for filename in filenames:
            if file_ext is None:
                for ext in SUPPORTED_EXTENSIONS:
                    if filename.endswith(ext):
                        files.append(os.path.join(root, filename))
                        break
            elif filename.endswith(file_ext):
                files.append(os.path.join(root, filename))
    return files

def mmap_merge_weights(files, file_ext):
    combined_state_dict = {}

    for file in files:
        if file_ext == ".pt" or file_ext == ".jit.pt":
            # Using mmap for PyTorch models
            with open(file, "rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                state_dict = torch.load(mm, map_location="cpu")
        elif file_ext == ".pth":
            # Using mmap for PyTorch model
            with open(file, "rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                state_dict = torch.load(mm, map_location="cpu")
        elif file_ext == ".pb" or file_ext == ".savedmodel":
            # TensorFlow models don’t benefit from mmap, load normally
            import tensorflow as tf
            model = tf.saved_model.load(file) if file_ext == ".savedmodel" else tf.compat.v1.saved_model.load(file)
            state_dict = model
        elif file_ext == ".onnx":
            # ONNX models don’t benefit from mmap, load normally
            import onnx
            model = onnx.load(file)
            state_dict = model
        elif file_ext == ".h5":
            # Keras models don’t benefit from mmap, load normally
            from tensorflow.keras.models import load_model
            model = load_model(file)
            state_dict = model
        elif file_ext == ".prototxt" or file_ext == ".param":
            # Caffe models don’t benefit from mmap, load normally
            pass
        elif file_ext == ".mlmodel":
            # CoreML models don’t benefit from mmap, load normally
            import coremltools
            model = coremltools.models.MLModel(file)
            state_dict = model
        elif file_ext == ".plan":
            # TensorRT models don’t benefit from mmap, load normally
            pass
        elif file_ext == ".xml":
            # OpenVINO models don’t benefit from mmap, load normally
            import openvino
            model = openvino.inference_engine.IENetwork(model=file)
            state_dict = model
        elif file_ext == ".safetensors":
            # Safetensors doesn’t benefit from mmap, load normally
            state_dict = load_file(file)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Combine the weights (add them or update them in the state dict)
        for key, value in state_dict.items():
            if key in combined_state_dict:
                combined_state_dict[key] += value
            else:
                combined_state_dict[key] = value
    
    return combined_state_dict


def merge_weights(files, file_ext, use_mmap=False):
    combined_state_dict = {}
    
    for file in files:
        if file_ext == ".pt" or file_ext == ".jit.pt":
            state_dict = torch.load(file, map_location="cpu")
        elif file_ext == ".pb" or file_ext == ".savedmodel":
            import tensorflow as tf
            model = tf.saved_model.load(file) if file_ext == ".savedmodel" else tf.compat.v1.saved_model.load(file)
            state_dict = model
        elif file_ext == ".onnx":
            import onnx
            model = onnx.load(file)
            state_dict = model
        elif file_ext == ".h5":
            from tensorflow.keras.models import load_model
            model = load_model(file)
            state_dict = model
        elif file_ext == ".prototxt" or file_ext == ".param":
            # Caffe Model loading
            pass
        elif file_ext == ".mlmodel":
            import coremltools
            model = coremltools.models.MLModel(file)
            state_dict = model
        elif file_ext == ".plan":
            # TensorRT model loading (not always straightforward, may require specific libraries)
            pass
        elif file_ext == ".xml":
            import openvino
            model = openvino.inference_engine.IENetwork(model=file)
            state_dict = model
        elif file_ext == ".safetensors":
            state_dict = load_file(file)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        for key, value in state_dict.items():
            if key in combined_state_dict:
                combined_state_dict[key] += value
            else:
                combined_state_dict[key] = value
    return combined_state_dict


def save_weights(state_dict, output_path, file_ext):
    if file_ext == ".pt" or file_ext == ".jit.pt":
        torch.save(state_dict, output_path)
    elif file_ext == ".pb":
        import tensorflow as tf
        # Save as TensorFlow model
        tf.saved_model.save(state_dict, output_path)
    elif file_ext == ".onnx":
        import onnx
        onnx.save(state_dict, output_path)
    elif file_ext == ".h5":
        from tensorflow.keras.models import save_model
        save_model(state_dict, output_path)
    elif file_ext == ".prototxt" or file_ext == ".param":
        # Caffe saving logic
        pass
    elif file_ext == ".mlmodel":
        import coremltools
        coremltools.models.MLModel(state_dict).save(output_path)
    elif file_ext == ".plan":
        # Saving TensorRT model
        pass
    elif file_ext == ".xml":
        import openvino
        # Save OpenVINO model
        openvino.save_model(state_dict, output_path)
    elif file_ext == ".safetensors":
        save_file(state_dict, output_path)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")


def main():
    parser = argparse.ArgumentParser(description="Merge model weights from multiple files into one.", add_help=False)
    parser.add_argument("-h", "--help", action="help", help="Show this help message and exit.")
    parser.add_argument("-i", "--input_dir", type=str, help="Input directory containing weight files.")
    parser.add_argument("-o", "--output_file", type=str, help="Output file name (without extension).")
    parser.add_argument("-t", "--model_type", type=str, help="Model type (e.g., .pth, .safetensors). Supported formats are: .pb, .pt, .onnx, .h5, .prototxt, .param, .savedmodel, .jit.pt, .mlmodel, .plan, .xml, .safetensors.")
    parser.add_argument("-m", "--mmap", action="store_true", help="Use memory-mapped files for loading .pth files.")
    args = parser.parse_args()

    if args.help:
        print("Usage: python model_merger.py [options]")
        print()
        print("Options:")
        print("  -h, --help            Show this help message and exit.")
        print("  -i, --input_dir       Input directory containing weight files.")
        print("  -o, --output_file     Output file name (without extension).")
        print("  -t, --model_type      Model type (e.g., .pth, .safetensors). Supported formats are: .pb, .pt, .onnx, .h5, .prototxt, .param, .savedmodel, .jit.pt, .mlmodel, .plan, .xml, .safetensors.")
        print("  -m, --mmap            Use memory-mapped files for loading .pth files.")
        sys.exit(0)

    dir_path = args.input_dir or os.getcwd()
    file_ext = args.model_type
    output_name = args.output_file

    if file_ext is None:
        files = get_file_list(dir_path)
    else:
        files = get_file_list(dir_path, file_ext)

    if not files:
        print("No files found with the specified extension.")
        sys.exit(1)

    # Use mmap if specified
    if args.mmap:
        state_dict = mmap_merge_weights(files, file_ext)
    else:
        state_dict = merge_weights(files, file_ext, args.mmap)
        
    output_dir = os.path.join(dir_path, "Smlmodel")
    os.makedirs(output_dir, exist_ok=True)

    if len(files) > 1:
        parent_folder = Path(dir_path).name
        output_path = os.path.join(output_dir, f"{output_name}_{parent_folder}{file_ext}")
    else:
        output_path = os.path.join(output_dir, f"{output_name}{file_ext}")

    save_weights(state_dict, output_path, file_ext)
    print(f"Combined weights saved at: {output_path}")


if __name__ == "__main__":
    main()