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
import org.matsim.core.controler.OutputDirectoryHierarchy;
import org.matsim.core.gbl.MatsimRandom;
import org.matsim.core.network.NetworkUtils;
import org.matsim.core.network.algorithms.NetworkCleaner;
import org.matsim.core.network.io.MatsimNetworkReader;
import org.matsim.core.population.PopulationUtils;
import org.matsim.core.population.io.PopulationReader;
import org.matsim.core.scenario.ScenarioUtils;

import java.io.File;
import java.util.*;

public class createScenarios_districtWorld {

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

        File newDir = new File("scenarios/districtWorlds/" + scenario_name);
        if (!newDir.exists()) {
            newDir.mkdirs();
        }

        Scenario sc = ScenarioUtils.createScenario(ConfigUtils.loadConfig("scenarios/berlin-v5.5-" + percentage_original + "pct/input/berlin-v5.5-" + percentage_original + "pct.config-local.xml"));
        Config conf = sc.getConfig();

        // CREATE CONFIG FILE
        System.out.println("1. CREATE CONFIG FILE");
        // 1. inputNetworkFile
        NetworkConfigGroup network_conf = conf.network();
        network_conf.setInputFile("network.xml");
        // 2. outputDirectory
        ControlerConfigGroup cont_conf = conf.controler();
        cont_conf.setOutputDirectory("scenarios/districtWorlds/" + scenario_name + "/output");
        cont_conf.setLastIteration(lastIteration);
        cont_conf.setOverwriteFileSetting(OutputDirectoryHierarchy.OverwriteFileSetting.deleteDirectoryIfExists);
        cont_conf.setRunId(scenario_name);
        // 3. inputPlansFile
        PlansConfigGroup plan_conf = conf.plans();
        plan_conf.setInputFile("plans.xml");
        // Save config
        ConfigWriter configWriter = new ConfigWriter(conf);
        configWriter.write("scenarios/districtWorlds/" + scenario_name + "/config.xml");

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
        int index = (int) Math.ceil(1.0 / 100.0 * distances.size()); //1.0
        Double max_distance = distances.get(index-1);

        for (Node node : network.getNodes().values()) {
            if ( NetworkUtils.getEuclideanDistance(center_node.getCoord(), node.getCoord()) > max_distance ){
                Node removed_node = network.removeNode(node.getId());
            }
        }

        System.out.println("Cleaning network ...\n");
        NetworkCleaner networkCleaner = new NetworkCleaner() ;
        networkCleaner.run(network) ;

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


        // CREATE SMALL POPULATION
        System.out.println("1. CREATE SMALL POPULATION FILE");
        PopulationReader popreader = new PopulationReader(sc);
        popreader.readFile("scenarios/berlin-v5.5-" + percentage_original + "pct/input/berlin-v5.5-" + percentage_original + "pct.plans.xml.gz");
        Population pop = sc.getPopulation();
        PopulationUtils.printPlansCount(pop);

        System.out.println("Delete outside population ...\n");
        ArrayList<Id<Person>> list_id_person_to_delete = new ArrayList<Id<Person>>();
        for (Person pers : pop.getPersons().values()) {
            for (Plan plan : pers.getPlans()){
                for (PlanElement plan_element : plan.getPlanElements()) {
                    if (plan_element instanceof Activity) {
                        Activity activity = (Activity) plan_element;
                        if ( ! network.getLinks().containsKey(activity.getLinkId()) ) {
                            list_id_person_to_delete.add(pers.getId());
                        }
                    }
                }
            }
        }

        for (Id<Person> person_id_to_delete : list_id_person_to_delete){
            pop.removePerson(person_id_to_delete);
        }

        System.out.println("Restructuring links of legs ...\n");
        for (Person pers : pop.getPersons().values()) {
            for (Plan plan : pers.getPlans()){
                for (PlanElement plan_element : plan.getPlanElements()) {
                    if (plan_element instanceof Leg) {
                        Leg leg = (Leg) plan_element;
                        //Activity activity_previous = PopulationUtils.getPreviousActivity(plan, leg);
                        //Activity activity_next = PopulationUtils.getNextActivity(plan, leg);
                        if (leg.getMode().equals("car")){
                            leg.setRoute(null);
                        }
                    }
                }
            }
        }
        // Save population file
        System.out.println("Writing population ...\n");
        PopulationUtils.writePopulation(pop, "scenarios/districtWorlds/" + scenario_name + "/plans.xml");
        PopulationUtils.printPlansCount(pop);

        // Save network file
        System.out.println("Writing network ...\n");
        new NetworkWriter(network).write("scenarios/districtWorlds/" + scenario_name + "/network.xml");
        System.out.println("DISTRICT WORLD CREATION DONE ...\n");
    }

}
