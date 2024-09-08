# The COAML pipeline as a surrogate model to approximate traffic flows

The code is structured as follows:
- **data:** containing training, validation, and testing data of different scenarios
- **generating:** code to create instances of different scenarios
  - generating new scenarios:
    - *againstsupervisedWorld_scenario.py:* The network consists of two areas: a working area and a home area. The two areas are only connected by a single road. Thus, each commuter has to use the connecting road, which is difficutl to learn from a pure deep learning perspective.
    - *entropy_scenarios.py:* Two different scenarios: The high entropy scenario generates a uniformly distribute population over a rando squared network. The low entropy scenario generates a population with home locations in the upper right corner, and work locations in the lower left corner.
    - *smallWorld_scenarios.py:* The network is a random newman_watts_strogatz_graph
    - *smallWorld_scenarios_roadpricing:* Similar to the smallWorld_scenarios but in the center of the network is a tolling area. Specifically, agents have to pay when using a road in this tolling area.
    - *smallWorld_scenarios_short.py:* Similar to the smallWorld_scenarios but the scenario only lasts for two hours. Thus, this scenario leads to more capacity conflicts and is less computational intense.
    - *squareWorld_scenarios_short.py:* The network is a squared scenario, according to https://github.com/arun1729/road-network
    - *squareWorld_scenarios_short_capacity:* Similar to squareWorld_scenarios_short but with road-specific capacity restrictions.
  - processing oracle data to training data:
    - *cutoutWorld_training_data.py:* Reading in a MATSim cutoutWorld scenario and generating json training data. Note, MATSim runs a complete Berlin scenario, and this python code splits the scenario into districts.
    - *cutoutWorldWithoutBoundary.py:* Reading in a MATSim cutoutWorld scenario and generating json training data, similar to cutoutWorld. When processing the data, we only consider trips that use roads with only one lane for at least 60% of their trip.
    - *districtWorld_training_data.py:* Reading in a MATSim districtWorld scenario and generating json training data.
    - *districtWorldArt_training_data.py:* Reading in a MATSim districtWorldArt scenario and generating json training data.
    - *smallWorld_short_training_data.py:* Reading in a MATSim smallWorld_short scenario and generating json training data.
    - *smallWorld_training_data.py:* Reading in a MATSim smallWorld scenario and generating json training data.
    - *sparseWorld_training_data.py:* Reading in a MATSim sparseWorld scenario and generating json training data.
    - *squareWorld_short_training_data.py:* Reading in a MATSim sparseWorld_short scenario and generating json training data.
- **pipelines:** code representing the COAML pipeline with an ML-layer, CO-layer and the learning framework
- **results:** saving directory for the result files
- **src:** supporting codes
- **visualizing:** code to visualize data, instances, and results
- **calculating_training_data.py:** generates training instances
- **compare_learning_iterations.py:** choosing the model iteration that performs best during training on a validation data set
- **evaluation.py:** code to evaluate the trained COAML pipeline
- **training.py:** code to train the COAML pipeline