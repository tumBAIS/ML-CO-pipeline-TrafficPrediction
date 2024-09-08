import sys
import os
import numpy as np
import pandas as pd
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


        minimum_start = scenario["minimum_start"]
        maximum_end = scenario["maximum_end"]
        half_time = (maximum_end + minimum_start) / 2

        vehicle_trips_to_work = e.events[(e.events["time"] > minimum_start) & (e.events["time"] < half_time)].groupby('vehicle')[["link", "time"]].apply(lambda x: x.values.tolist()).to_dict()
        vehicle_trips_to_home = e.events[(e.events["time"] > half_time) & (e.events["time"] < maximum_end)].groupby('vehicle')[["link", "time"]].apply(lambda x: x.values.tolist()).to_dict()

        mapping = []
        o_d_pairs = []
        commodities = []
        o_d_pairs_starting_time = []
        o_d_pairs_ending_time = []

        edge_flow_times = np.array(scenario["link_length"]) / np.array(scenario["link_freespeed"])

        commodity = 0
        for vehicle_trips in [vehicle_trips_to_work, vehicle_trips_to_home]:
            for vehicle_trip in vehicle_trips.values():
                for (link, time) in vehicle_trip:
                    mapping.append({"time": time, "link": link, "commodity": commodity})
                start_node = scenario["link_from"][int(vehicle_trip[0][0])]
                end_node = scenario["link_to"][int(vehicle_trip[-1][0])]
                o_d_pairs.append((start_node, end_node))
                o_d_pairs_starting_time.append(vehicle_trip[0][1])
                o_d_pairs_ending_time.append(vehicle_trip[-1][1] + edge_flow_times[int(link)])
                commodities.append(commodity)
                commodity += 1

        mapping = pd.DataFrame(mapping)

        scenario["o_d_pairs"] = o_d_pairs
        scenario["commodities"] = commodities
        scenario["o_d_pairs_starting_time"] = o_d_pairs_starting_time
        scenario["o_d_pairs_ending_time"] = o_d_pairs_ending_time
        scenario["solution_time"] = np.array(mapping.time)
        scenario["solution_link"] = np.array(mapping.link)
        scenario["solution_commodity"] = np.array(mapping.commodity)


    dir_instances = get_directory_instances(args)
    create_directory(f"{dir_instances}/")
    save_json(scenario, f"{dir_instances}/s-{args.seed}{x}.json")
