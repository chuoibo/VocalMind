import yaml
import os
import torch
import json
from pathlib import Path
import logging

def warm_up_model(device, model):
    dummy_input = torch.zeros(1, 16000).to(device) 
    for _ in range(3):  
        with torch.no_grad():
            _ = model(dummy_input).logits


def read_yaml_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)  
        return data
    except FileNotFoundError:
        logging.info(f"Error: The file {file_path} was not found.")
    except yaml.YAMLError as exc:
        logging.info(f"Error parsing YAML file: {exc}")
    except Exception as e:
        logging.info(f"An error occurred: {e}")


def make_directory(path):
    try:
        os.makedirs(path, exist_ok=True)  
        logging.info(f"Directory '{path}' created successfully.")
    except Exception as e:
        logging.info(f"Error creating directory '{path}': {e}")


def write_to_txt_file(text, file_path):
    try:
        with open(file_path, 'w') as file:
            file.write(text)
        logging.info(f"Text successfully written to {file_path}")
    except Exception as e:
        logging.info(f"An error occurred: {e}")
    return file_path


def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def load_json(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        logging.info(f"Error: The file {file_path} was not found.")
    except json.JSONDecodeError:
        logging.info(f"Error: The file {file_path} is not a valid JSON file.")
    except Exception as e:
        logging.info(f"An unexpected error occurred: {e}")


def save_json(data, file_path, indent=4):
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=indent)
            logging.info(f"Data successfully saved to {file_path}")
    except Exception as e:
        logging.info(f"An error occurred while saving to {file_path}: {e}")


def find_files(directory_path, type_file):
    directory = Path(directory_path)
    return list(directory.rglob(f"*.{type_file}"))