import sys
import os
import numpy as np
import pandas as pd
import collections
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
main_directory = "/".join(SCRIPT_DIR.split("/")[:-2])
sys.path.append(main_directory)
from surrogate.visualizing.event_reader import EventReader
from surrogate.src.util import load_json, save_json, get_directory_instances, create_directory, get_scenario_acronym, get_scenario_directory_acronym, Graph

def generate_training_data(args, x="", with_target=True):

    # Read in scenario
    scenario = load_json(f'../matsim-berlin/scenarios/{args.simulation}/{get_scenario_acronym(args)}/scenario.json')
    # Read in output
    if with_target:
        e = EventReader(path_input=f'../matsim-berlin/scenarios/{args.simulation}/{get_scenario_acronym(args)}/output{x}/{get_scenario_acronym(args)}.output_events.xml.gz')

        link_counts = {key: 0 for key in range(len(scenario["links_id"]))}
        for key, value in e.link_counts.to_dict("dict")["count"].items():
            link_counts[int(key)] = value
        scenario["link_counts"] = link_counts

    dir_instances = get_directory_instances(args)
    create_directory(f"{dir_instances}/")
    save_json(scenario, f"{dir_instances}/s-{args.seed}{x}.json")
