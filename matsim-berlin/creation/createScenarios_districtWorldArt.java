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
import org.matsim.core.network.algorithms.NetworkCleaner;
import org.matsim.core.network.io.MatsimNetworkReader;
import org.matsim.core.population.PopulationUtils;
import org.matsim.core.scenario.ScenarioUtils;

import java.io.File;
import java.util.*;

public class createScenarios_districtWorldArt {

    public static void main(String[] args) {

        int percentage_original = 1;
        double percentage_new = 1.0;
        int seed_new = 0;
        int seed = 0;
        int lastIteration = 3;

        for (int i = 0; i < args.length; i++) {
            if ("-po".equals(args[i])) {
                percentage_original = Integer.parseInt(args[++i]);
            } else if ("-pn".equals(args[i])) {
                percentage_new = Double.parseDouble(args[++i]);
            } else if ("-sn".equals(args[i])) {
                seed_new = Integer.parseInt(args[++i]);
            } else if ("-s".equals(args[i])) {
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

        String scenario_name = "po-" + percentage_original + "_pn-" + percentage_new + "_sn-" + seed_new + "_s-" + seed;

        File newDir = new File("scenarios/districtWorldsArt/" + scenario_name);
        if (!newDir.exists()) {
            newDir.mkdirs();
        }

        Scenario sc = ScenarioUtils.createScenario(ConfigUtils.loadConfig("./scenarios/equil/config.xml"));
        Config conf = sc.getConfig();

        // CREATE CONFIG FILE
        System.out.println("1. CREATE CONFIG FILE");
        // 1. inputNetworkFile
        NetworkConfigGroup network_conf = conf.network();
        network_conf.setInputFile("network.xml");
        // 2. outputDirectory
        ControlerConfigGroup cont_conf = conf.controler();
        cont_conf.setOutputDirectory("scenarios/districtWorldsArt/" + scenario_name + "/output");
        cont_conf.setLastIteration(lastIteration);
        cont_conf.setOverwriteFileSetting(OutputDirectoryHierarchy.OverwriteFileSetting.deleteDirectoryIfExists);
        cont_conf.setRunId(scenario_name);
        // 3. inputPlansFile
        PlansConfigGroup plan_conf = conf.plans();
        plan_conf.setInputFile("plans.xml");
        // QSim
        QSimConfigGroup qsim_conf = conf.qsim();
        qsim_conf.setStartTime(8 * 3600);
        qsim_conf.setEndTime(20 * 3600);
        // Save config
        ConfigWriter configWriter = new ConfigWriter(conf);
        configWriter.write("scenarios/districtWorldsArt/" + scenario_name + "/config.xml");

        // CREATE SMALL NETWORK
        System.out.println("1. CREATE SMALL NETWORK FILE");
        System.out.println("Reading network ...\n");
        MatsimNetworkReader networkReader = new MatsimNetworkReader(sc.getNetwork());
        networkReader.readFile("original-input-data/berlin-v5.5-network.xml.gz");
        Network network = sc.getNetwork();
        Link[] original_links = NetworkUtils.getSortedLinks(network);
        Node[] original_nodes = NetworkUtils.getSortedNodes(network);

        // Delete PT network
        List<Id<Link>> link_ids = NetworkUtils.getLinkIds(Arrays.asList(original_links));
        for (Id<Link> link_id : link_ids) {
            if (link_id.toString().contains("pt_")) {
                Link removed_link = network.removeLink(link_id);
            }
        }

        List<Node> network_nodes = Arrays.asList(NetworkUtils.getSortedNodes(network));
        Node center_node = network_nodes.get(rand.nextInt(network_nodes.size()));

        ArrayList<Double> distances = new ArrayList<Double>();
        for (Node node : network.getNodes().values()) {
            distances.add(NetworkUtils.getEuclideanDistance(center_node.getCoord(), node.getCoord()));
        }
        // Get percentile
        Collections.sort(distances);
        int index = (int) Math.ceil(0.6 / 100.0 * distances.size()); //1.0
        Double max_distance = distances.get(index-1);

        for (Node node : network.getNodes().values()) {
            if ( NetworkUtils.getEuclideanDistance(center_node.getCoord(), node.getCoord()) > max_distance ){
                Node removed_node = network.removeNode(node.getId());
            }
        }

        System.out.println("Cleaning network ...\n");
        NetworkCleaner networkCleaner = new NetworkCleaner() ;
        networkCleaner.run(network) ;

        Node[] updated_nodes = NetworkUtils.getSortedNodes(network);

        // CREATE SMALL POPULATION
        System.out.println("CREATE POPULATION FILE");
        Population population = PopulationUtils.createPopulation(conf, network);
        PopulationFactory populationFactory = population.getFactory();

        Random random = new Random();
        for (int idx = 0; idx < 50; idx++) { //100
            // Create person
            Person person = populationFactory.createPerson(Id.create(idx, Person.class));
            // Create and add attributes
            person.getAttributes().putAttribute( "subpopulation", "person");
            // Create plan
            Plan plan = PopulationUtils.createPlan(person);
            // Create activities and legs
            int randomIndexHome = random.nextInt(updated_nodes.length);
            Node home_node = updated_nodes[randomIndexHome];
            Activity activity_home1 = PopulationUtils.createActivityFromCoord("h", home_node.getCoord());
            activity_home1.setEndTime(9 * 3600);
            plan.addActivity(activity_home1);
            Leg leg_to_work = PopulationUtils.createLeg("car");
            plan.addLeg(leg_to_work);

            int randomIndexWork = random.nextInt(updated_nodes.length);
            Node work_node = updated_nodes[randomIndexWork];
            Activity activity_work = PopulationUtils.createActivityFromCoord("w", work_node.getCoord());
            activity_work.setEndTime(16 * 3600);
            plan.addActivity(activity_work);
            Leg leg_to_home = PopulationUtils.createLeg("car");
            plan.addLeg(leg_to_home);
            Activity activity_home2 = PopulationUtils.createActivityFromCoord("h", home_node.getCoord());
            plan.addActivity(activity_home2);
            // Add plan to person
            person.addPlan(plan);
            // Add person to population
            population.addPerson(person);
        }

        // Fill pt network again
        for (Node node_i : original_nodes) {
            if (node_i.toString().contains("pt_")) {
                network.addNode(node_i);
            }
        }
        for (Link link_i : original_links) {
            if (link_i.toString().contains("pt_")) {
                network.addLink(link_i);
            }
        }

        // Save population file
        System.out.println("Writing population ...\n");
        PopulationUtils.writePopulation(population, "scenarios/districtWorldsArt/" + scenario_name + "/plans.xml");
        PopulationUtils.printPlansCount(population);

        // Save network file
        System.out.println("Writing network ...\n");
        new NetworkWriter(network).write("scenarios/districtWorldsArt/" + scenario_name + "/network.xml");
        System.out.println("DISTRICT WORLD CREATION DONE ...\n");
    }

}
