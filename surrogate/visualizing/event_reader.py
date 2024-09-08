import matsim
import pandas as pd


class PlanReader:
    def __init__(self, path_input):
        self.file_path_input = path_input
        self.home_activities_location = []
        self.work_activities_location = []
        self.read_plans()

    def get_home_work_locations(self):
        for person, plan in self.plans:
            relevant_activities = filter(lambda e: (e.tag == 'activity') and (('home' in e.attrib['type']) or ('work' in e.attrib['type'])), plan)
            for activity in relevant_activities:
                x_location = float(activity.attrib["x"])
                y_location = float(activity.attrib["y"])
                if "home" in activity.attrib["type"]:
                    self.home_activities_location.append([x_location, y_location])
                elif "work" in activity.attrib["type"]:
                    self.work_activities_location.append([x_location, y_location])

    def get_o_d_pairs(self, mapping_link_old_id, graph):
        o_d_link_pairs = []
        for person, plan in self.plans:
            car_plans = list(filter(lambda e: ((e.tag == 'leg') and (e.attrib['mode'] == "car")) or ((e.tag == "activity") and (e.attrib["type"] == "car interaction")), plan))
            for car_plan_index, car_plan in enumerate(car_plans):
                if car_plan.tag == "leg" and car_plan.attrib["mode"] == "car":
                    o_d_link_pairs.append((car_plans[car_plan_index - 1].attrib["link"], car_plans[car_plan_index + 1].attrib["link"]))
        o_d_pairs = [(graph.links[mapping_link_old_id[int(link_start)]].to_node, graph.links[mapping_link_old_id[int(link_end)]].from_node) for link_start, link_end in o_d_link_pairs]
        return o_d_pairs

    def read_plans(self):
        self.plans = matsim.plan_reader(self.file_path_input, selectedPlansOnly=True)


class TripReader:
    def __init__(self, path_input):
        self.file_path_input = path_input
        self.read_trips()

    def read_trips(self):
        print("START READING TRIPS...")
        self.trips = pd.read_csv(self.file_path_input, compression='gzip', sep=";")

    def get_o_d_pairs(self, mapping_link_old_id, graph):
        relevant_trips = self.trips[(self.trips["main_mode"] == "car")]  # | (self.trips["main_mode"] == "ride")
        relevant_trips_start_link = relevant_trips["start_link"]
        relevant_trips_stop_link = relevant_trips["end_link"]
        o_d_pairs = [(graph.links[mapping_link_old_id[link_start]].to_node, graph.links[mapping_link_old_id[link_end]].from_node) for link_start, link_end in zip(relevant_trips_start_link, relevant_trips_stop_link)]
        return o_d_pairs


class EventReader:
    def __init__(self, path_input):
        self.file_path_input = path_input
        self.read_events()

    def read_events(self):
        print("START READING EVENTS...")

        self.events = pd.DataFrame(matsim.event_reader(self.file_path_input, types='entered link'))
        self.events = self.events[~self.events['link'].astype(str).str.startswith('pt_')]
        self.events = self.events[~self.events['link'].astype(str).str.startswith('freight_')]
        self.events = self.events[~self.events['vehicle'].astype(str).str.startswith('freight_')]
        self.events["link"] = pd.to_numeric(self.events["link"])
        self.events = self.events.drop(columns=['type'])


