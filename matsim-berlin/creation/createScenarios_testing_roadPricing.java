import org.matsim.api.core.v01.Coord;
import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.network.Link;
import org.matsim.api.core.v01.network.Network;
import org.matsim.api.core.v01.network.Node;
import org.matsim.api.core.v01.population.*;
import org.matsim.contrib.roadpricing.RoadPricingConfigGroup;
import org.matsim.contrib.roadpricing.RoadPricingSchemeImpl;
import org.matsim.contrib.roadpricing.RoadPricingUtils;
import org.matsim.contrib.roadpricing.RoadPricingWriterXMLv1;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.config.groups.ControlerConfigGroup;
import org.matsim.core.config.groups.NetworkConfigGroup;
import org.matsim.core.config.groups.PlansConfigGroup;
import org.matsim.core.controler.OutputDirectoryHierarchy;
import org.matsim.core.gbl.MatsimRandom;
import org.matsim.core.network.NetworkUtils;
import org.matsim.core.population.PopulationUtils;
import org.matsim.core.scenario.ScenarioUtils;

import java.io.File;
import java.util.Random;

public class createScenarios_testing_roadPricing {

    public static void main(String[] args) {

        int seed = 1;
        boolean withroadpricing = true;

        Random rand = new Random(seed);
        MatsimRandom.reset(seed);

        String scenario_name = "";
        if (withroadpricing) {
            scenario_name = "withroadpricing";
        }
        else {
            scenario_name = "original";
        }

        File newDir = new File("scenarios/testing/" + scenario_name);
        if (!newDir.exists()) {
            newDir.mkdirs();
        }
        Config config = ConfigUtils.loadConfig("scenarios/equil/config.xml") ;
        Scenario scenario = ScenarioUtils.createScenario(config) ;

        System.out.println("1. CHANGE CONFIG FILE");
        // set network import file
        NetworkConfigGroup network_conf = config.network();
        network_conf.setInputFile("network.xml");
        // set output directory
        ControlerConfigGroup cont_conf = config.controler();
        cont_conf.setOutputDirectory("scenarios/testing/" + scenario_name + "/output");
        cont_conf.setLastIteration(3);
        cont_conf.setOverwriteFileSetting(OutputDirectoryHierarchy.OverwriteFileSetting.deleteDirectoryIfExists);
        cont_conf.setRunId(scenario_name);
        // Set plans import file
        PlansConfigGroup plan_conf = config.plans();
        plan_conf.setInputFile("plans.xml");

        System.out.println("Creating network ...\n");
        Network network = NetworkUtils.createNetwork();
        Node start_node = NetworkUtils.createAndAddNode(network, Id.create(0, Node.class), new Coord(0.0, 0.0));
        Node intermediate1node = NetworkUtils.createAndAddNode(network, Id.create(1, Node.class), new Coord(20.0, 20.0));;
        Node intermediate2node = NetworkUtils.createAndAddNode(network, Id.create(2, Node.class), new Coord(50.0, 50.0));
        Node intermediate3node = NetworkUtils.createAndAddNode(network, Id.create(3, Node.class), new Coord(80.0, 80.0));
        Node end_node = NetworkUtils.createAndAddNode(network, Id.create(4, Node.class), new Coord(100.0, 100.0));

        Link short_link_start_inter1 = NetworkUtils.createAndAddLink(network, Id.create(0, Link.class), start_node, intermediate1node, 100, 10, 1, 1);
        Link short_link_inter1_inter2 = NetworkUtils.createAndAddLink(network, Id.create(1, Link.class), intermediate1node, intermediate2node, 100, 10, 1, 1);
        Link short_link_inter2_inter3 = NetworkUtils.createAndAddLink(network, Id.create(2, Link.class), intermediate2node, intermediate3node, 100, 10, 1, 1);
        Link short_link_inter3_end = NetworkUtils.createAndAddLink(network, Id.create(3, Link.class), intermediate3node, end_node, 100, 10, 1, 1);

        Link cheap_link_start_inter1 = NetworkUtils.createAndAddLink(network, Id.create(4, Link.class), start_node, intermediate1node, 200, 10, 1, 1);
        Link cheap_link_inter1_inter2 = NetworkUtils.createAndAddLink(network, Id.create(5, Link.class), intermediate1node, intermediate2node, 200, 10, 1, 1);
        Link cheap_link_inter2_inter3 = NetworkUtils.createAndAddLink(network, Id.create(6, Link.class), intermediate2node, intermediate3node, 200, 10, 1, 1);
        Link cheap_link_inter3_end = NetworkUtils.createAndAddLink(network, Id.create(7, Link.class), intermediate3node, end_node, 200, 5, 10, 1);

        Link home_link_end_inter3 = NetworkUtils.createAndAddLink(network, Id.create(8, Link.class), end_node, intermediate3node, 100, 10, 1, 1);
        Link home_link_inter3_inter2 = NetworkUtils.createAndAddLink(network, Id.create(9, Link.class), intermediate3node, intermediate2node, 100, 10, 1, 1);
        Link home_link_inter2_inter1 = NetworkUtils.createAndAddLink(network, Id.create(10, Link.class), intermediate2node, intermediate1node, 100, 10, 1, 1);
        Link home_link_inter1_start = NetworkUtils.createAndAddLink(network, Id.create(11, Link.class), intermediate1node, start_node, 100, 10, 1, 1);


        System.out.println("Creating population ... \n");
        Population pop = PopulationUtils.createPopulation(config, network);
        PopulationFactory pop_fac = pop.getFactory();
        Person person = pop_fac.createPerson(Id.create(0, Person.class));
        person.getAttributes().putAttribute( config.plans().getSubpopulationAttributeName(), "person");
        Plan plan = PopulationUtils.createPlan(person);
        Activity activity_home = PopulationUtils.createAndAddActivityFromCoord(plan, "h", start_node.getCoord());
        activity_home.setEndTime(1000);
        PopulationUtils.createAndAddLeg(plan, "car");
        Activity activity_work = PopulationUtils.createAndAddActivityFromCoord(plan, "w", end_node.getCoord());
        activity_work.setEndTime(50000);
        PopulationUtils.createAndAddLeg(plan, "car");
        Activity activity_home2 = PopulationUtils.createAndAddActivityFromCoord(plan, "h", start_node.getCoord());
        person.addPlan(plan);
        pop.addPerson(person);

        if (withroadpricing) {
            System.out.println("CREATE ROAD PRICING SCHEME ... \n");
            RoadPricingSchemeImpl roadPricingScheme = RoadPricingUtils.addOrGetMutableRoadPricingScheme(scenario);
            RoadPricingUtils.setType(roadPricingScheme, "distance");
            RoadPricingUtils.setName(roadPricingScheme, "distance toll");
            RoadPricingUtils.setDescription(roadPricingScheme, "distance toll");

            RoadPricingUtils.addLink(roadPricingScheme, short_link_start_inter1.getId());
            RoadPricingUtils.addLink(roadPricingScheme, short_link_inter1_inter2.getId());
            RoadPricingUtils.addLink(roadPricingScheme, short_link_inter2_inter3.getId());
            RoadPricingUtils.addLink(roadPricingScheme, short_link_inter3_end.getId());

            RoadPricingUtils.addLinkSpecificCost(roadPricingScheme, short_link_start_inter1.getId(), 0.0, 108000.0, 10);
            RoadPricingUtils.addLinkSpecificCost(roadPricingScheme, short_link_inter1_inter2.getId(), 0.0, 108000.0, 10);
            RoadPricingUtils.addLinkSpecificCost(roadPricingScheme, short_link_inter2_inter3.getId(), 0.0, 108000.0, 10);
            RoadPricingUtils.addLinkSpecificCost(roadPricingScheme, short_link_inter3_end.getId(), 0.0, 108000.0, 10);


            RoadPricingWriterXMLv1 roadPricingWriterXMLv1 = new RoadPricingWriterXMLv1(roadPricingScheme);
            roadPricingWriterXMLv1.writeFile("scenarios/testing/" + scenario_name + "/tolling_scheme.xml");

            RoadPricingConfigGroup roadPricingConfigGroup = RoadPricingUtils.createConfigGroup();
            roadPricingConfigGroup.setTollLinksFile("tolling_scheme.xml");
            config.addModule(roadPricingConfigGroup);
        }

        System.out.println("Writing config ...\n");
        ConfigUtils.writeConfig(config, "scenarios/testing/" + scenario_name + "/config.xml");

        System.out.println("Writing population ...\n");
        PopulationUtils.writePopulation(pop, "scenarios/testing/" + scenario_name + "/plans.xml");

        System.out.println("Writing network ...\n");
        NetworkUtils.writeNetwork(network, "scenarios/testing/" + scenario_name + "/network.xml");

        System.out.println("FINISH!");

    }
}
