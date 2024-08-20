from flask import Flask, request
from flask_socketio import SocketIO, emit

import hashlib
import zipfile
import os
import io
import base64
import numpy as np
import onnx
import torch
from onnx import numpy_helper

from src.utils.execute_utilities import convert_onnx_to_pytorch, make_global_model_client_ready, return_onnx_global_model, save_new_global_model
from src.models.models import ModelTrainer

app = Flask(__name__)
# CORS(app)
# CORS(app,resources={r"/*":{"origins":"*"}})
socketio = SocketIO(app,cors_allowed_origins="*")

client_list = [] #List of client IDs
client_federation_round_update_monitor = [] #Dictionary of client IDs and their connection URLs
client_federated_round_model_updates = [] #List of client model updates
client_socket_session_ids = [] #List of client socket session IDs

device = "cpu" # device type for the global model
global_model = ModelTrainer().to(device) #Global model

is_global_model_is_created = False
federation_round = 0
webfl_client_id = 0
MAX_FEDERATION_ROUNDS = 1001

def create_global_model():
    global global_model
    model = global_model

def zip_folder(folder_path):
    # Create an in-memory zip file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Add file to the zip
                zip_file.write(file_path, os.path.relpath(file_path, folder_path))
    zip_buffer.seek(0)  # Rewind the buffer to the beginning
    return zip_buffer.getvalue()

def calculate_checksum(data):
    return hashlib.sha256(data).hexdigest()

def aggregate_global_model(onnx_model, a_weights):
    weights = torch.load(a_weights)

    onnx_weights = {}
    for initializer in onnx_model.graph.initializer:
        name = initializer.name

        if name in weights:
            # Convert the PyTorch tensor to a numpy array
            new_weight = weights[name].numpy()

            # Update the initializer's data with the new weight
            initializer.raw_data = numpy_helper.from_array(new_weight).raw_data

            # Store the updated weight
            onnx_weights[name] = new_weight

    return onnx_model

@socketio.on('connect')
def handle_connect():
    global webfl_client_id
    print(f"Client connected: {request.sid}")
    client_list.append(webfl_client_id)
    payload = {'status': 'Connected to server', 'webfl_client_id': webfl_client_id}
    emit('client_connected', {'data': payload})
    webfl_client_id += 1

@socketio.on('connection_message_from_client')
def handle_message(data):
    print(f"Received message from client: {data}")

    global client_socket_session_ids
    client_socket_session_ids.append(request.sid)
    payload = {'data': request.sid}
    emit('connected_message_from_server', payload)

@socketio.on('request_global_model')
def handle_request_global_model(data):
    print(f"Client {data} is requesting the global model")
    global is_global_model_is_created, global_model, federation_round

    if not is_global_model_is_created:
        create_global_model()
        is_global_model_is_created = True

    model_path = make_global_model_client_ready(global_model, federation_round)
    
    zip_data = zip_folder(model_path)
    checksum = calculate_checksum(zip_data)
    # Encode the binary data to Base64
    zip_file = base64.b64encode(zip_data).decode('utf-8')
    payload = {'data': zip_file, 'checksum': checksum}

    # Debugging: Print length and a snippet of the Base64 string
    print(f"Base64 Encoded Length: {len(zip_file)}")
    print(f"Base64 Encoded Snippet: {zip_file[:100]}")

    emit('global_model_from_server', payload)

socketio.on('client_training_completed')
def continue_training(data):
    global federation_round, client_federated_round_model_updates, client_federation_round_update_monitor, MAX_FEDERATION_ROUNDS, global_model
    print(f"Client has completed initial training")

    while (federation_round <= MAX_FEDERATION_ROUNDS):
        webfl_client_id = data['webfl_client_id']
        loss = data['loss']
        weights = data['modelWeights']

        client_federated_round_model_updates.append({'webfl_client_id': webfl_client_id, 'loss': loss, 'weights': weights, 'fl_round': federation_round})

        tmp_client_list = []
        for i in client_federated_round_model_updates:
            tmp_client_list.append(i['webfl_client_id'])

            if len(client_federation_round_update_monitor[federation_round]) == len(tmp_client_list):
                weight_buffers = []
                for i in client_federated_round_model_updates:
                    weight_buffers.append(i['weights'])
                stack_buffer = np.stack(weight_buffers, axis=0)
                global_model_weights = np.mean(stack_buffer, axis=0).astype(np.uint8)
                global_onnx_model = onnx.load(return_onnx_global_model(federation_round))
                new_global_model = aggregate_global_model(global_onnx_model, global_model_weights)
                federation_round += 1
                save_new_global_model(new_global_model, federation_round)
                payload = {weights: global_model_weights}
                emit('broadcast_global_model', {'data': payload}, broadcast=True, include_self=False)
            else:
                print("Waiting for all clients to finish training")

    
    if federation_round >= MAX_FEDERATION_ROUNDS:
        global_model = convert_onnx_to_pytorch(federation_round)
        print("Federation rounds have been completed")
        emit('disconnect', {'data': 'Federation rounds have been completed'}, broadcast=True)

if __name__ == '__main__':
    socketio.run(app, debug=True)
