import yaml
import easydict


def load_yaml(file_path: str):
    assert file_path.endswith('.yaml') or file_path.endswith('.yml'), \
        "File must be a YAML file"

    # Open the YAML file
    with open(file_path, 'r') as file:
        # Load the YAML content into a Python object
        yaml_data = yaml.safe_load(file)

    return easydict.EasyDict(yaml_data)
