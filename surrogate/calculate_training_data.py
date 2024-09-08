import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
main_directory = "/".join(SCRIPT_DIR.split("/")[:-1])
sys.path.append(main_directory)
from surrogate.generating.cutoutWorld_training_data import generate_training_data as cutoutWorld_generate_training_data
from surrogate.generating.squareWorld_short_training_data import generate_training_data as squareWorldShort_generate_training_data
from surrogate.generating.cutoutWorldWithoutBoundary import generate_training_data as cutoutWorldWithoutBoundary_generate_training_data
from surrogate.generating.districtWorld_training_data import generate_training_data as districtWorld_generate_training_data
from surrogate.generating.districtWorldArt_training_data import generate_training_data as districtWorldArt_generate_training_data
from surrogate.src import config


if __name__ == '__main__':
    args = config.parser.parse_args()
    generate_training_data = {"cutoutWorlds": cutoutWorld_generate_training_data, "cutoutWorldsCap": cutoutWorld_generate_training_data,
                              "cutoutWorldsSpeed": cutoutWorld_generate_training_data,
                              "squareWorlds_short": squareWorldShort_generate_training_data,
                              "cutoutWorldsWithoutBoundary": cutoutWorldWithoutBoundary_generate_training_data,
                              "districtWorlds": districtWorld_generate_training_data,
                              "districtWorldsArt": districtWorldArt_generate_training_data}
    # Process simulation results and save context information which is the input to the pipeline
    generate_training_data[args.simulation](args)
