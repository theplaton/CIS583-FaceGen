import torch
import json
import os


def get_project_dir():
    return os.path.dirname(os.path.abspath(__file__))

def read_configs():
    # Specify the path to your JSON file
    json_file_path = os.path.join(get_project_dir(), 'config.json')

    # Read the JSON file into a dictionary
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        
    return data

def get_model_path_by_name(model_name):
    return os.path.join(read_configs()['model_dir_rel'], model_name)


def resolve_device():
    # Check for CUDA and MPS availability
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()

    # Set the device based on availability
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Print the selected device
    print(f"Using device: {device}")
    
    return device

def check_mps():
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print (x)
    else:
        print ("MPS device not found.")