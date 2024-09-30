import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.network.Link;
import org.matsim.api.core.v01.network.Network;
import org.matsim.api.core.v01.network.NetworkWriter;
import org.matsim.api.core.v01.population.*;
import org.matsim.core.config.Config;
import org.matsim.core.config.groups.ControlerConfigGroup;
import org.matsim.core.config.groups.NetworkConfigGroup;
import org.matsim.core.config.groups.PlansConfigGroup;
import org.matsim.core.config.groups.QSimConfigGroup;
import org.matsim.core.controler.OutputDirectoryHierarchy;
import org.matsim.core.network.io.MatsimNetworkReader;
import org.matsim.core.population.algorithms.PlansFilterByLegMode;
import org.matsim.core.scenario.ScenarioUtils;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.population.io.PopulationReader;
import org.matsim.core.population.PopulationUtils;
import org.matsim.core.config.ConfigWriter;

import java.io.File;
import org.matsim.core.gbl.MatsimRandom;


public class createScenarios_cutoutWorldCap {

    public static void main(String[] args) {

        int percentage_original = 1;
        double percentage_new = 0.1;
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

        MatsimRandom.reset(seed);

        String scenario_name = "po-" + percentage_original + "_pn-" + percentage_new + "_sn-" + seed_new;

        File newDir = new File("scenarios/cutoutWorldsCap/" + scenario_name);
        if (!newDir.exists()) {
            newDir.mkdirs();
        }

        Scenario sc = ScenarioUtils.createScenario(ConfigUtils.loadConfig("scenarios/berlin-v5.5-" + percentage_original + "pct/input/berlin-v5.5-" + percentage_original + "pct.config-local.xml"));
        Config conf = sc.getConfig();

        System.out.println("1. CHANGE CONFIG FILE");
        // InputNetworkFile
        NetworkConfigGroup network_conf = conf.network();
        network_conf.setInputFile("network.xml");
        // OutputDirectory
        ControlerConfigGroup cont_conf = conf.controler();
        cont_conf.setOutputDirectory("scenarios/cutoutWorldsCap/" + scenario_name + "/output");
        cont_conf.setLastIteration(lastIteration);
        cont_conf.setOverwriteFileSetting(OutputDirectoryHierarchy.OverwriteFileSetting.deleteDirectoryIfExists);
        cont_conf.setRunId(scenario_name);
        // InputPlansFile
        PlansConfigGroup plan_conf = conf.plans();
        plan_conf.setInputFile("plans.xml");
        // Qsim
        QSimConfigGroup qsim_conf = conf.qsim();
        double original_capacity_factor = qsim_conf.getFlowCapFactor();
        double new_capacity_factor = original_capacity_factor * ((0.1 * percentage_new) / percentage_original);
        qsim_conf.setFlowCapFactor(new_capacity_factor);
        qsim_conf.setStorageCapFactor(new_capacity_factor);
        // Save config file
        ConfigWriter configWriter = new ConfigWriter(conf);
        configWriter.write("scenarios/cutoutWorldsCap/" + scenario_name + "/config.xml");

        System.out.println("1. CREATE SMALL POPULATION FILE");
        PopulationReader popreader = new PopulationReader(sc);
        popreader.readFile("scenarios/berlin-v5.5-" + percentage_original + "pct/input/berlin-v5.5-" + percentage_original + "pct.plans.xml.gz");
        Population pop = sc.getPopulation();
        PopulationUtils.printPlansCount(pop);

        // Delete all PT Plans - so there only remain car plans
        PlansFilterByLegMode plans_filter = new PlansFilterByLegMode("pt", PlansFilterByLegMode.FilterType.removeAllPlansWithMode);
        plans_filter.run(pop);

        PopulationUtils.sampleDown(pop, percentage_new);
        PopulationUtils.writePopulation(pop, "scenarios/cutoutWorldsCap/" + scenario_name + "/plans.xml");
        PopulationUtils.printPlansCount(pop);

        System.out.println("2. CREATE SMALL NETWORK FILE");
        System.out.println("Reading network ...\n");
        MatsimNetworkReader networkReader = new MatsimNetworkReader(sc.getNetwork());
        networkReader.readFile("original-input-data/berlin-v5.5-network.xml.gz");
        Network network = sc.getNetwork();

        // Adapt the capacity of highways to normal streets
        for (Link link : network.getLinks().values()) {
            double link_cap = link.getCapacity();
            if (link_cap > 1000) {
                link.setCapacity(1000);
            }
        }

        // Save network file
        System.out.println("Writing network ...\n");
        new NetworkWriter(network).write("scenarios/cutoutWorldsCap/" + scenario_name + "/network.xml");

        System.out.println("FINISH!");
    }
}
