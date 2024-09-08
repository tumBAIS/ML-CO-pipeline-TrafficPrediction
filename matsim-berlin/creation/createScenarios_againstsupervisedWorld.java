import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import org.matsim.api.core.v01.Coord;
import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.network.Link;
import org.matsim.api.core.v01.network.Network;
import org.matsim.api.core.v01.network.NetworkWriter;
import org.matsim.api.core.v01.network.Node;
import org.matsim.api.core.v01.population.*;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.config.ConfigWriter;
import org.matsim.core.config.groups.ControlerConfigGroup;
import org.matsim.core.config.groups.NetworkConfigGroup;
import org.matsim.core.config.groups.PlansConfigGroup;
import org.matsim.core.config.groups.QSimConfigGroup;
import org.matsim.core.controler.OutputDirectoryHierarchy;
import org.matsim.core.gbl.MatsimRandom;
import org.matsim.core.network.NetworkUtils;
import org.matsim.core.population.PopulationUtils;
import org.matsim.core.scenario.ScenarioUtils;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

public class createScenarios_againstsupervisedWorld {

    public static void main(String[] args) {

        int seed = 1;
        int lastIteration = 3;

        for (int i = 0; i < args.length; i++) {
            if ("-s".equals(args[i])) {
                seed = Integer.parseInt(args[++i]);
            } else if ("-li".equals(args[i])) {
                lastIteration = Integer.parseInt(args[++i]);
            } else {
                System.err.println("Error: unrecognized argument");
                System.exit(1);
            }
        }

        Random rand = new Random(seed);
        MatsimRandom.reset(seed);

        String scenario_name = "s-" + seed;

        File newDir = new File("scenarios/againstsupervisedWorld/" + scenario_name);
        if (!newDir.exists()) {
            newDir.mkdirs();
        }

        Scenario sc_equil = ScenarioUtils.createScenario(ConfigUtils.loadConfig("./scenarios/equil/config.xml"));
        Config conf = sc_equil.getConfig();

        System.out.println("1. CHANGE CONFIG FILE");
        // InputNetworkFile
        NetworkConfigGroup network_conf = conf.network();
        network_conf.setInputFile("network.xml");
        // OutputDirectory
        ControlerConfigGroup cont_conf = conf.controler();
        cont_conf.setOutputDirectory("scenarios/againstsupervisedWorld/" + scenario_name + "/output");
        cont_conf.setLastIteration(lastIteration);
        cont_conf.setOverwriteFileSetting(OutputDirectoryHierarchy.OverwriteFileSetting.deleteDirectoryIfExists);
        cont_conf.setRunId(scenario_name);
        // InputPlansFile
        PlansConfigGroup plan_conf = conf.plans();
        plan_conf.setInputFile("plans.xml");
        // QSim
        QSimConfigGroup qsim_conf = conf.qsim();
        qsim_conf.setStartTime(8 * 3600);
        qsim_conf.setEndTime(10 * 3600);
        // Save the config file
        ConfigWriter configWriter = new ConfigWriter(conf);
        configWriter.write("scenarios/againstsupervisedWorld/" + scenario_name + "/config.xml");


        System.out.println("2. READ IN JSON SCENARIO");
        JSONObject json_data = null;
        try {
            Object o = new JSONParser().parse(new FileReader("scenarios/againstsupervisedWorld/" + scenario_name + "/scenario.json"));
            json_data = (JSONObject) o;
        } catch (IOException | ParseException e) {
            e.printStackTrace();
        }
        JSONArray nodes_x = (JSONArray) json_data.get("nodes_x");
        JSONArray nodes_y = (JSONArray) json_data.get("nodes_y");
        JSONArray nodes_id = (JSONArray) json_data.get("nodes_id");
        JSONArray link_from = (JSONArray) json_data.get("link_from");
        JSONArray link_to = (JSONArray) json_data.get("link_to");
        JSONArray links_id = (JSONArray) json_data.get("links_id");

        JSONArray link_length = (JSONArray) json_data.get("link_length");
        JSONArray link_freespeed = (JSONArray) json_data.get("link_freespeed");
        JSONArray link_capacity = (JSONArray) json_data.get("link_capacity");
        JSONArray link_permlanes = (JSONArray) json_data.get("link_permlanes");

        JSONArray work_x = (JSONArray) json_data.get("work_x");
        JSONArray work_y = (JSONArray) json_data.get("work_y");
        JSONArray home_x = (JSONArray) json_data.get("home_x");
        JSONArray home_y = (JSONArray) json_data.get("home_y");
        JSONArray go_to_work = (JSONArray) json_data.get("go_to_work");
        JSONArray go_to_home = (JSONArray) json_data.get("go_to_home");

        System.out.println("3. CREATE NETWORK FILE");
        Network network = NetworkUtils.createNetwork(conf);
        ArrayList<Node> node_list = new ArrayList<>();
        for (int idx = 0; idx < nodes_id.size(); idx++) {
            node_list.add(NetworkUtils.createAndAddNode(network, Id.create(String.valueOf(nodes_id.get(idx)), Node.class), new Coord((Double) nodes_x.get(idx), (Double) nodes_y.get(idx))));
        }
        for (int idx = 0; idx < links_id.size(); idx++) {
            Node node_start = node_list.get(((Long) link_from.get(idx)).intValue());
            Node node_end = node_list.get(((Long) link_to.get(idx)).intValue());
            NetworkUtils.createAndAddLink(network,
                    Id.create(String.valueOf(links_id.get(idx)), Link.class),
                    node_start, node_end,
                    (Double) link_length.get(idx),
                    (Double) link_freespeed.get(idx),
                    (Double) link_capacity.get(idx),
                    (Double) link_permlanes.get(idx));
        }
        // Save network file
        NetworkWriter networkwriter = new NetworkWriter(network);
        networkwriter.write("scenarios/againstsupervisedWorld/" + scenario_name + "/network.xml");

        System.out.println("3. CREATE POPULATION FILE");
        Population population = PopulationUtils.createPopulation(conf, network);
        PopulationFactory populationFactory = population.getFactory();

        for (int idx = 0; idx < work_x.size(); idx++) {
            // Create person
            Person person = populationFactory.createPerson(Id.create(idx, Person.class));
            // Create and add attributes
            person.getAttributes().putAttribute( "subpopulation", "person");
            // Create plan
            Plan plan = PopulationUtils.createPlan(person);
            // Create activities and legs
            Activity activity_home1 = PopulationUtils.createActivityFromCoord("h", new Coord((Double) home_x.get(idx), (Double) home_y.get(idx)));
            activity_home1.setEndTime((Double) go_to_work.get(idx));
            plan.addActivity(activity_home1);
            Leg leg_to_work = PopulationUtils.createLeg("car");
            plan.addLeg(leg_to_work);
            Activity activity_work = PopulationUtils.createActivityFromCoord("w", new Coord((Double) work_x.get(idx), (Double) work_y.get(idx)));
            activity_work.setEndTime((Double) go_to_home.get(idx));
            plan.addActivity(activity_work);
            Leg leg_to_home = PopulationUtils.createLeg("car");
            plan.addLeg(leg_to_home);
            Activity activity_home2 = PopulationUtils.createActivityFromCoord("h", new Coord((Double) home_x.get(idx), (Double) home_y.get(idx)));
            plan.addActivity(activity_home2);
            // Add plan to person
            person.addPlan(plan);
            // Add person to population
            population.addPerson(person);
        }
        // Save population file
        PopulationUtils.writePopulation(population, "scenarios/againstsupervisedWorld/" + scenario_name + "/plans.xml");

        System.out.println("FINISH!");

    }



}
