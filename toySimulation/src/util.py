import json
import numpy as np
from surrogate.src.util import get_scenario_acronym


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def read_in_scenario(args):
    scenario = load_json(f'../matsim-berlin/scenarios/{args.simulation}/{get_scenario_acronym(args)}/scenario.json')
    #scenario = load_json(args.scenario_path + "/scenario.json")
    return scenario
