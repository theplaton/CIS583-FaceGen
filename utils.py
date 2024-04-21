import torch
import json
import os
import time
import sys


def get_project_dir():
    return os.path.dirname(os.path.abspath(__file__))

# resolves config based on abs or rel paths
def resolve_config(path):
    configs = read_configs()
    if path[-3:] == "abs":
        return configs[path]
    elif path[-3:] == "rel":
        return os.path.join(get_project_dir(), configs[path])
    else:
        raise AttributeError("no such path option (last 3 chars in path), only 'rel' and 'abs' allowed")


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

# method to print progress and estimate remaining time        
def print_progress_with_time(idx, total_imgs, start_time, interval):
    if idx % interval != 0 and idx != total_imgs - 1:
        return  # Skip printing if not at an interval or the last iteration

    current_time = time.time()
    elapsed_time = current_time - start_time
    avg_time_per_img = elapsed_time / (idx + 1)
    remaining_imgs = total_imgs - (idx + 1)
    remaining_time = remaining_imgs * avg_time_per_img / 60
    
    progress = (idx + 1) / total_imgs
    bar_length = 50
    filled_length = int(bar_length * progress)
    bar = '=' * filled_length + '-' * (bar_length - filled_length)
    
    sys.stdout.write(f'\rProgress: [{bar}] {progress*100:.2f}%  Time remaining: {remaining_time:.2f} mins')
    sys.stdout.flush()
    
    
# method to print progress and estimate remaining time        
def print_training_progress_with_time(idx, epoch, total_imgs, total_epochs, start_time, interval):
    if idx % interval != 0 and idx != total_imgs - 1:
        return  # Skip printing if not at an interval or the last iteration

    current_time = time.time()
    elapsed_time = current_time - start_time
    avg_time_per_img = elapsed_time / (idx + 1 + epoch * total_imgs)
    avg_time_per_epoch = avg_time_per_img * total_imgs
    remaining_imgs = total_imgs - (idx + 1)
    remaining_epochs = total_epochs - (epoch + 1)
    remaining_time = (remaining_imgs * avg_time_per_img + remaining_epochs * total_imgs)/ 60
    
    progress = (idx + 1) / total_imgs
    bar_length = 50
    filled_length = int(bar_length * progress)
    bar = '=' * filled_length + '-' * (bar_length - filled_length)
    
    sys.stdout.write(f'\rProgress: [{bar}] {progress*100:.2f}%  --- Time remaining: {remaining_time:.2f} mins  --- Average time/epoch: {avg_time_per_epoch:.2f} mins')
    sys.stdout.flush()