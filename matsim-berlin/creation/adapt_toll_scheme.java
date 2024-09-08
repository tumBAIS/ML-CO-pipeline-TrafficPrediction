import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.network.Link;
import org.matsim.api.core.v01.network.Network;
import org.matsim.contrib.roadpricing.*;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.network.NetworkUtils;
import org.matsim.core.scenario.ScenarioUtils;

import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class adapt_toll_scheme {

    public static void main(String[] args) {

        int seed = 1;
        double toll = 5;

        for (int i = 0; i < args.length; i++) {
            if ("-s".equals(args[i])) {
                seed = Integer.parseInt(args[++i]);
            } else if ("-t".equals(args[i])) {
                toll = Double.parseDouble(args[++i]);
            } else {
                System.err.println("Error: unrecognized argument");
                System.exit(1);
            }
        }

        String scenario_name = "s-" + seed;

        Config config = ConfigUtils.loadConfig("scenarios/smallWorlds_pricing/" + scenario_name + "/config.xml") ;
        Scenario scenario = ScenarioUtils.loadScenario(config) ;

        System.out.println("Reading network ...\n");
        Network network = scenario.getNetwork();

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
                    RoadPricingUtils.addLinkSpecificCost(roadPricingScheme, network_link.getId(), 0.0, 108000.0, toll);
                }
            }
        }

        RoadPricingWriterXMLv1 roadPricingWriterXMLv1 = new RoadPricingWriterXMLv1(roadPricingScheme);
        roadPricingWriterXMLv1.writeFile("scenarios/smallWorlds_pricing/" + scenario_name + "/tolling_scheme[" + toll + "].xml");

    }
}
