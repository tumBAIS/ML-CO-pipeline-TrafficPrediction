import org.json.simple.JSONArray;
import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.network.Link;
import org.matsim.api.core.v01.population.Population;
import org.matsim.contrib.roadpricing.*;
import org.matsim.contrib.roadpricing.RoadPricingUtils;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.controler.OutputDirectoryHierarchy;
import org.matsim.core.gbl.MatsimRandom;
import org.matsim.core.scenario.ScenarioUtils;
import org.matsim.core.config.groups.NetworkConfigGroup;
import org.matsim.core.config.groups.PlansConfigGroup;
import org.matsim.core.config.groups.ControlerConfigGroup;
import org.matsim.api.core.v01.network.Network;
import org.matsim.core.network.NetworkUtils;
import org.json.simple.JSONObject;
import org.json.simple.parser.*;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.io.File;

import org.matsim.core.population.PopulationUtils;


public class createScenarios_smallWorld_roadPricing {

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

        File newDir = new File("scenarios/smallWorlds_pricing/" + scenario_name);
        if (!newDir.exists()) {
            newDir.mkdirs();
        }

        Config config = ConfigUtils.loadConfig("scenarios/smallWorlds/" + scenario_name + "/config.xml") ;
        Scenario scenario = ScenarioUtils.loadScenario(config) ;

        System.out.println("1. CHANGE CONFIG FILE");
        // set network import file
        NetworkConfigGroup network_conf = config.network();
        network_conf.setInputFile("network.xml");
        // set output directory
        ControlerConfigGroup cont_conf = config.controler();
        cont_conf.setOutputDirectory("scenarios/smallWorlds_pricing/" + scenario_name + "/output");
        cont_conf.setLastIteration(lastIteration);
        cont_conf.setOverwriteFileSetting(OutputDirectoryHierarchy.OverwriteFileSetting.deleteDirectoryIfExists);
        cont_conf.setRunId(scenario_name);
        // Set plans import file
        PlansConfigGroup plan_conf = config.plans();
        plan_conf.setInputFile("plans.xml");

        System.out.println("Reading network ...\n");
        Network network = scenario.getNetwork();

        System.out.println("Reading population ... \n");
        Population pop = scenario.getPopulation();

        System.out.println("CREATE ROAD PRICING SCHEME ... \n");
        RoadPricingSchemeImpl roadPricingScheme = RoadPricingUtils.addOrGetMutableRoadPricingScheme(scenario);
        RoadPricingUtils.setType(roadPricingScheme, "distance");
        RoadPricingUtils.setName(roadPricingScheme, "distance toll");
        RoadPricingUtils.setDescription(roadPricingScheme, "distance toll");

        System.out.println("READ IN JSON SCENARIO ...");
        JSONObject json_data = null;
        try {
            Object o = new JSONParser().parse(new FileReader("scenarios/smallWorlds_pricing/" + scenario_name + "/scenario.json"));
            json_data = (JSONObject) o;
        } catch (IOException | ParseException e) {
            e.printStackTrace();
        }
        JSONArray toll_links = (JSONArray) json_data.get("toll_links");

        Link[] inner_street_links = NetworkUtils.getSortedLinks(network);
        List<Link> inner_street_links_list = Arrays.asList(inner_street_links);
        for (int i = 0; i < toll_links.size(); i++) {
            String toll_link_idx = ((Long) toll_links.get(i)).toString();
            for (int j = 0; j < inner_street_links_list.size(); j++) {
                Link network_link = inner_street_links_list.get(j);
                if (network_link.getId().toString().equals(toll_link_idx)) {
                    RoadPricingUtils.addLink(roadPricingScheme, network_link.getId());
                    RoadPricingUtils.addLinkSpecificCost(roadPricingScheme, network_link.getId(), 0.0, 108000.0, 10);
                }
            }
        }

        RoadPricingWriterXMLv1 roadPricingWriterXMLv1 = new RoadPricingWriterXMLv1(roadPricingScheme);
        roadPricingWriterXMLv1.writeFile("scenarios/smallWorlds_pricing/" + scenario_name + "/tolling_scheme.xml");

        RoadPricingConfigGroup roadPricingConfigGroup = RoadPricingUtils.createConfigGroup();
        roadPricingConfigGroup.setTollLinksFile("/tolling_scheme.xml");
        config.addModule(roadPricingConfigGroup);

        System.out.println("Writing config ...\n");
        ConfigUtils.writeConfig(config, "scenarios/smallWorlds_pricing/" + scenario_name + "/config.xml");

        System.out.println("Writing population ...\n");
        PopulationUtils.writePopulation(pop, "scenarios/smallWorlds_pricing/" + scenario_name + "/plans.xml");

        System.out.println("Writing network ...\n");
        NetworkUtils.writeNetwork(network, "scenarios/smallWorlds_pricing/" + scenario_name + "/network.xml");

        System.out.println("FINISH!");

    }



}
