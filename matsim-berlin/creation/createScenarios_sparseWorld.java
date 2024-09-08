import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.population.*;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.config.ConfigWriter;
import org.matsim.core.config.groups.ControlerConfigGroup;
import org.matsim.core.config.groups.NetworkConfigGroup;
import org.matsim.core.config.groups.PlansConfigGroup;
import org.matsim.core.controler.OutputDirectoryHierarchy;
import org.matsim.core.population.PopulationUtils;
import org.matsim.core.population.io.PopulationReader;
import org.matsim.core.scenario.ScenarioUtils;
import org.matsim.core.network.io.MatsimNetworkReader;
import org.matsim.core.network.algorithms.NetworkCleaner;
import org.matsim.api.core.v01.network.Network;
import org.matsim.api.core.v01.network.NetworkWriter;
import org.matsim.core.network.NetworkUtils;
import org.matsim.api.core.v01.network.Link;
import org.matsim.api.core.v01.network.Node;
import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.population.Activity;
import org.matsim.api.core.v01.Coord;

import java.util.*;
import java.io.File;

import org.matsim.core.gbl.MatsimRandom;

public class createScenarios_sparseWorld {

    public static void main(String[] args) throws InterruptedException {
        int percentage_original = 1;
        int seed = 0;
        int lastIteration = 3;

        for (int i = 0; i < args.length; i++) {
            if ("-po".equals(args[i])) {
                percentage_original = Integer.parseInt(args[++i]);
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

        String scenario_name = "po-" + percentage_original + "_s-" + seed;

        File newDir = new File("scenarios/sparseWorlds/" + scenario_name);
        if (!newDir.exists()) {
            newDir.mkdirs();
        }

        Scenario sc = ScenarioUtils.createScenario(ConfigUtils.loadConfig("scenarios/berlin-v5.5-" + percentage_original + "pct/input/berlin-v5.5-" + percentage_original + "pct.config-local.xml"));
        Config conf_new = sc.getConfig();

        // CREATE CONFIG FILE
        System.out.println("1. CREATE CONFIG FILE");
        // 1. inputNetworkFile
        NetworkConfigGroup network_conf = conf_new.network();
        network_conf.setInputFile("network.xml");
        // 2. controller
        ControlerConfigGroup cont_conf = conf_new.controler();
        cont_conf.setOutputDirectory("scenarios/sparseWorlds/" + scenario_name + "/output");
        cont_conf.setLastIteration(lastIteration);
        cont_conf.setOverwriteFileSetting(OutputDirectoryHierarchy.OverwriteFileSetting.deleteDirectoryIfExists);
        cont_conf.setRunId(scenario_name);
        // 3. inputPlansFile
        PlansConfigGroup plan_conf = conf_new.plans();
        plan_conf.setInputFile("plans.xml");
        // Save config
        ConfigWriter configWriter = new ConfigWriter(conf_new);
        configWriter.write("scenarios/sparseWorlds/" + scenario_name + "/config.xml");


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

        Link[] street_links = NetworkUtils.getSortedLinks(network);
        List<Link> street_links_list = Arrays.asList(street_links);
        for (int i = 0; i < street_links_list.size(); i++) {
            Link link = street_links_list.get(i);
            Coord from_node_coord = link.getFromNode().getCoord();
            Coord to_node_coord = link.getToNode().getCoord();
            if ( from_node_coord.getX() < 4580000 | from_node_coord.getX() > 4613000 | from_node_coord.getY() < 5807000 | from_node_coord.getY() > 5835000 |
                    to_node_coord.getX() < 4580000 | to_node_coord.getX() > 4613000 | to_node_coord.getY() < 5807000 | to_node_coord.getY() > 5835000 ){
                Link removed_link = network.removeLink(link.getId());
            }
        }

        int num_remaining_links_after1Clean = NetworkUtils.getSortedLinks(network).length;


        System.out.println("Cleaning network 1...\n");
        NetworkCleaner firstCleaner = new NetworkCleaner() ;
        firstCleaner.run(network) ;

        Link[] inner_street_links = NetworkUtils.getSortedLinks(network);
        List<Link> innner_street_links_list = Arrays.asList(inner_street_links);
        Collections.shuffle(innner_street_links_list, rand);
        for (int i = 0; i < innner_street_links_list.size(); i++) {
            Link link = innner_street_links_list.get(i);
            boolean keep = false;
            if (link.getNumberOfLanes() > 1) {   //1 -> 6800
                keep = true;
            }
            if (!keep) {
                Link removed_link = network.removeLink(link.getId());
            }
        }

        System.out.println("Cleaning network 2...\n");
        NetworkCleaner secondCleaner = new NetworkCleaner() ;
        secondCleaner.run(network) ;

        int network_size = network.getLinks().size();
        while(network_size > 2000) {
            Link[] while_inner_street_links = NetworkUtils.getSortedLinks(network);
            List<Link> while_inner_street_links_list = Arrays.asList(while_inner_street_links);
            Collections.shuffle(while_inner_street_links_list, rand);
            Link link = while_inner_street_links_list.get(0);

            Network artificial_network = NetworkUtils.createNetwork();
            for (Node artificial_node : NetworkUtils.getSortedNodes(network)) {
                NetworkUtils.createAndAddNode(artificial_network, artificial_node.getId(), artificial_node.getCoord());
            }
            for (Link artificial_link : NetworkUtils.getSortedLinks(network)) {
                NetworkUtils.createAndAddLink(artificial_network,
                        artificial_link.getId(), artificial_link.getFromNode(), artificial_link.getToNode(),
                        artificial_link.getLength(), artificial_link.getFreespeed(), artificial_link.getCapacity(), artificial_link.getNumberOfLanes());
            }
            int num_before_cleaning = artificial_network.getLinks().size();
            Link removed_artificial_link = artificial_network.removeLink(link.getId());
            NetworkCleaner XCleanerFirst = new NetworkCleaner();
            XCleanerFirst.run(artificial_network);

            if (((float) artificial_network.getLinks().size() / (float) num_before_cleaning) > 0.90) {
                Link removed_link = network.removeLink(link.getId());
                NetworkCleaner XCleanerSecond = new NetworkCleaner();
                XCleanerSecond.run(network);
            }
            network_size = network.getLinks().size();
            System.out.println("Cleaning network X...remaining " + network_size);
        }

        double max_x = 0;
        double min_x = 100000000;
        double max_y = 0;
        double min_y = 100000000;

        for (Node node : network.getNodes().values()){
            Coord coord = node.getCoord();
            if (coord.getX() > max_x) {
                max_x = coord.getX();
            }
            if (coord.getX() < min_x) {
                min_x = coord.getX();
            }
            if (coord.getY() > max_y) {
                max_y = coord.getY();
            }
            if (coord.getY() < min_y) {
                min_y = coord.getY();
            }
        }


        int num_remaining_links_after2Clean = NetworkUtils.getSortedLinks(network).length;

        // CREATE SMALL POPULATION
        System.out.println("1. CREATE SMALL POPULATION FILE");
        PopulationReader popreader = new PopulationReader(sc);
        popreader.readFile("scenarios/berlin-v5.5-" + percentage_original + "pct/input/berlin-v5.5-" + percentage_original + "pct.plans.xml.gz");
        Population pop = sc.getPopulation();
        PopulationUtils.printPlansCount(pop);
        float remaining_population = (float) num_remaining_links_after2Clean / num_remaining_links_after1Clean;
        System.out.println("We keep " + remaining_population + " from the original population");
        PopulationUtils.sampleDown(pop, remaining_population);

        System.out.println("Delete outside population ...\n");
        ArrayList<Id<Person>> list_id_person_to_delete = new ArrayList<Id<Person>>();
        for (Person pers : pop.getPersons().values()) {
            for (Plan plan : pers.getPlans()){
                for (PlanElement plan_element : plan.getPlanElements()) {
                    if (plan_element instanceof Activity) {
                        Activity activity = (Activity) plan_element;
                        Coord coord_activity = activity.getCoord();
                        if ( coord_activity.getX() < min_x | coord_activity.getX() > max_x | coord_activity.getY() < min_y | coord_activity.getY() > max_y) {
                            list_id_person_to_delete.add(pers.getId());
                        }
                    }
                }
            }
        }

        for (Id<Person> person_id_to_delete : list_id_person_to_delete){
            pop.removePerson(person_id_to_delete);
        }

        System.out.println("Restructuring population ...\n");
        System.out.println("Restructuring links of activities ...\n");
        for (Person pers : pop.getPersons().values()) {
            for (Plan plan : pers.getPlans()){
                for (PlanElement plan_element : plan.getPlanElements()) {
                    if (plan_element instanceof Activity) {
                        Activity activity = (Activity) plan_element;
                        if (activity.getType().equals("pt interaction")){
                            continue;
                        }
                        Coord coord_activity = activity.getCoord();
                        Link link_activity = NetworkUtils.getNearestLink(network, coord_activity);
                        activity.setLinkId(link_activity.getId());
                    }
                }
            }
        }
        System.out.println("Restructuring links of legs ...\n");
        for (Person pers : pop.getPersons().values()) {
            for (Plan plan : pers.getPlans()){
                for (PlanElement plan_element : plan.getPlanElements()) {
                    if (plan_element instanceof Leg) {
                        Leg leg = (Leg) plan_element;
                        Activity activity_previous = PopulationUtils.getPreviousActivity(plan, leg);
                        Activity activity_next = PopulationUtils.getNextActivity(plan, leg);
                        if (leg.getMode().equals("car")){
                            leg.setRoute(null);
                        }
                        else if (leg.getMode().equals("freight")){
                            leg.setRoute(null);
                        }
                        else if (leg.getMode().equals("walk")){
                            leg.setRoute(null);
                        }
                        else if (leg.getMode().equals("bicycle")){
                            leg.setRoute(null);
                        }
                        else if (leg.getMode().equals("ride")){
                            leg.setRoute(null);
                        }
                        else {
                            Route route = leg.getRoute();
                            route.setStartLinkId(activity_previous.getLinkId());
                            route.setEndLinkId(activity_next.getLinkId());
                        }
                    }
                }
            }
        }
        System.out.println("Writing population ...\n");
        PopulationUtils.writePopulation(pop, "scenarios/sparseWorlds/" + scenario_name + "/plans.xml");
        PopulationUtils.printPlansCount(pop);

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
        System.out.println("Writing network ...\n");
        new NetworkWriter(network).write("scenarios/sparseWorlds/" + scenario_name + "/network.xml");

        System.out.println("SPARSE WORLD CREATION DONE ...\n");


    }

}
