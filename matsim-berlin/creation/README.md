# Creation of MATSim scenarios

The files serve to create stylized MATSim scenarios:
- **createScenarios_againstsupervisedWorld:** Reading in the json scenario from againstsupervisedWorld and creating a MATSim scenario.
- **createScenarios_cutoutWorld:** Samples down the population of the Berlin scenario.
- **createScenarios_cutoutWorldCap:** Samples down the population of the Berlin scenario. Sets the maximum street capacity to 1000.
- **createScenarios_cutoutWorldSpeed:** Samples down the population of the Berlin scenario. Sets a consistent street capacity of 1000 and a freespeed of 10.
- **createScenarios_districtWorld:** Each scenario has a random center node and only considers roads that have a maximum distance from this center node
- **createScenarios_districtWorldArt:** Similar to districtWorld but also generating an artificial population.
- **createScenarios_smallWorld:** Reading in the json scenario from smallWorld and creating a MATSim scenario.
- **createScenarios_smallWorld_roadPricing:** Loads a MATSim scenario from smallWorld and adds roadPricing.
- **createScenarios_smallWorld_short:** Reading in the json scenario from smallWorld_short and creating a MATSim scenario.
- **createScenarios_sparseWorld:** Sparsing the Berlin scenario and only keeping a skeleton of the network as a MATSim scenario.
- **createScenarios_squareWorld_short:** Reading in the json scenario from squareWorld_short and creating a MATSim scenario.
- **createScenarios_squareWorld_short_capacitated:** Reading in the json scenario from squareWorld_short_capacitated and creating a MATSim scenario.
- **createScenarios_testing_roadPricing:** toy MATSim scenario to test the impact of roadpricing on the routing results.