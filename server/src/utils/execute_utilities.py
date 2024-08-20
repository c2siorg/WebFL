import os
import torch
import torch.onnx
import onnx
from onnx2pytorch import ConvertModel
import tempfile
from onnxruntime.training import artifacts
from copy import deepcopy
import subprocess

TMP_FILE_DIRECTORY = tempfile.gettempdir()
model_save_path = os.path.join(TMP_FILE_DIRECTORY, "server")

def convert_pytorch_to_onnx(model, input_size, onnx_path):
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    # model.eval()
    dummy_input = torch.randn(*input_size)
    torch.onnx.export(model, dummy_input, onnx_path, do_constant_folding=False, training=torch.onnx.TrainingMode.TRAINING, export_params=True)
    # sanatize_tf_function_parameters(onnx_path)
    print(f"Model converted to ONNX and saved to {onnx_path}")

def save_new_global_model(global_model, federated_round):
    global model_save_path
    onnx.save(global_model, model_save_path+ "/global_model_"+str(federated_round)+".onnx")

def generate_onnx_training_artifacts(onnx_model_path, global_model, artifacts_path):
    os.makedirs(os.path.dirname(artifacts_path), exist_ok=True)
    print(artifacts_path)
    onnx_model = onnx.load(onnx_model_path)

    requires_grad = [name for name, param in global_model.named_parameters() if param.requires_grad]
    frozen_params = [name for name, param in global_model.named_parameters() if not param.requires_grad]

    artifacts.generate_artifacts(
        onnx_model,
        optimizer=artifacts.OptimType.SGD,
        loss=artifacts.LossType.MSELoss,
        artifact_directory=artifacts_path,
        requires_grad=requires_grad,
        frozen_params=frozen_params)
    
def convert_onnx_to_pytorch(federated_round):
    # Load ONNX model
    onnx_model = onnx.load(model_save_path+ "/global_model_"+str(federated_round)+".onnx")

    # Convert ONNX model to PyTorch
    pytorch_model = ConvertModel(onnx_model)
    print(f"ONNX model converted back to PyTorch")
    return pytorch_model

def return_onnx_global_model(federated_round):
    return os.path.join(TMP_FILE_DIRECTORY, "server/global_model_"+str(federated_round)+".onnx")

def make_global_model_client_ready(global_model, federated_round):
    tmp_global_model = deepcopy(global_model)
    global model_save_path

    convert_pytorch_to_onnx(tmp_global_model, (1, 10), model_save_path+ "/global_model_"+str(federated_round)+".onnx")
    generate_onnx_training_artifacts(model_save_path+"/global_model_"+str(federated_round)+".onnx", global_model, model_save_path+"/global_model_"+str(federated_round)+"_onnx_artifacts/")
    return model_save_path+"/global_model_"+str(federated_round)+"_onnx_artifacts"

# had to change in onnx2pytorch/convert/model.py, onnx2pytorch/convert/layer.py