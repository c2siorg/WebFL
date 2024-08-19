import os
import torch
import torch.onnx
import onnx
from onnx import helper
from onnx_tf.backend import prepare
import tempfile
from copy import deepcopy
import subprocess
import tensorflowjs as tfjs
from onnx2pytorch import ConvertModel

TMP_FILE_DIRECTORY = tempfile.gettempdir()

def convert_pytorch_to_onnx(model, input_size, onnx_path):
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    model.eval()
    dummy_input = torch.randn(*input_size)
    torch.onnx.export(model, dummy_input, onnx_path, export_params=True)
    sanatize_tf_function_parameters(onnx_path)
    print(f"Model converted to ONNX and saved to {onnx_path}")

# Had to use as a workaround to fix the issue with the ONNX model not being able to be converted to TensorFlow (ex: KeyError: 'onnx::Gemm_0')
def sanatize_tf_function_parameters(onnx_model_path):
    onnx_model = onnx.load(onnx_model_path)
    name_map = {"onnx::Gemm_0": "onnx__Gemm_0"}

    # Initialize a list to hold the new inputs
    new_inputs = []

    # Iterate over the inputs and change their names if needed
    for inp in onnx_model.graph.input:
        if inp.name in name_map:
            # Create a new ValueInfoProto with the new name
            new_inp = helper.make_tensor_value_info(name_map[inp.name],
                                                    inp.type.tensor_type.elem_type,
                                                    [dim.dim_value for dim in inp.type.tensor_type.shape.dim])
            new_inputs.append(new_inp)
        else:
            new_inputs.append(inp)

    # Clear the old inputs and add the new ones
    onnx_model.graph.ClearField("input")
    onnx_model.graph.input.extend(new_inputs)

    # Go through all nodes in the model and replace the old input name with the new one
    for node in onnx_model.graph.node:
        for i, input_name in enumerate(node.input):
            if input_name in name_map:
                node.input[i] = name_map[input_name]

    # Save the renamed ONNX model
    onnx.save(onnx_model, onnx_model_path)

def convert_onnx_to_tf(onnx_path, tf_path):
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_path)
    print(f"ONNX model converted to TensorFlow and saved to {tf_path}")

def convert_tf_to_tfjs(tf_path, tfjs_path):
    command = [
      "tfjs_graph_converter",
      "--input_format=tf_saved_model",
      "--output_format=tfjs_graph_model",
      tf_path,
      tfjs_path
    ]

    try:
        subprocess.run(command, check=True)
        print("Conversion successful!")
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed with error: {e}")

def convert_tfjs_to_tf(tfjs_path, tf_path):
    command = [
      "tfjs_graph_converter",
      "--output_format=tf_saved_model",
      tfjs_path,
      tf_path
    ]

    try:
        subprocess.run(command, check=True)
        print("Conversion successful!")
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed with error: {e}")

def convert_tf_to_onnx(tf_path, onnx_path):
    command = [
      "python",
      "-m"
      "tf2onnx.convert",
      "--saved-model",
      tf_path,
      "--output",
      onnx_path
    ]

    try:
        subprocess.run(command, check=True)
        print("Conversion successful!")
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed with error: {e}")

def convert_onnx_to_pytorch(onnx_path):
    # Load ONNX model
    onnx_model = onnx.load(onnx_path)

    # Convert ONNX model to PyTorch
    pytorch_model = ConvertModel(onnx_model)
    print(f"ONNX model converted back to PyTorch")
    return pytorch_model

def make_global_model_client_ready(global_model, federated_round):
    tmp_global_model = deepcopy(global_model)
    model_save_path = os.path.join(TMP_FILE_DIRECTORY, "server")
    convert_pytorch_to_onnx(tmp_global_model, (1, 10), model_save_path+ "/global_model_"+str(federated_round)+".onnx")
    convert_onnx_to_tf(model_save_path+"/global_model_"+str(federated_round)+".onnx", model_save_path+"/global_model_"+str(federated_round)+"_model_tf")
    convert_tf_to_tfjs(model_save_path+"/global_model_"+str(federated_round)+"_model_tf", model_save_path+"/global_model_"+str(federated_round)+"_model_tfjs")
    return model_save_path+"/global_model_"+str(federated_round)+"_model_tfjs"


def make_client_model_global_ready(client_model_path, federated_round):
    model_save_path = os.path.join(TMP_FILE_DIRECTORY, "server")
    convert_tfjs_to_tf(client_model_path, model_save_path+"/client_updated_model_"+str(federated_round)+"_model_tf")
    convert_tf_to_onnx(model_save_path+"/client_updated_model_"+str(federated_round)+"_model_tf", model_save_path+"/client_updated_model_"+str(federated_round)+".onnx")
    torch_model = convert_onnx_to_pytorch(model_save_path+"/client_updated_model_"+str(federated_round)+".onnx")
    return torch_model



# had to change in onnx2pytorch/convert/model.py, onnx2pytorch/convert/layer.py, tfjs_graph_converter/util.py