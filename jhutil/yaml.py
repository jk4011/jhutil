
import sys
sys.path.append("./jhutil")

from copy import deepcopy
from easydict import EasyDict
import importlib
from kitsu import logger, options, utils

# get parse argument as yaml file and parse the yaml file into a dictionary.
get_yaml_config = options.get_config

# convert a dictionary to a instant object.
instantiate_from_config = utils.instantiate_from_config



