from utils.utilities import get_source_date_type
from casatasks import importuvfits
import argparse
import os


parser = argparse.ArgumentParser(description='Transfrom uvf data to ms format.')
parser.add_argument('--data_file', type=str, help='path to the data file.')
args = parser.parse_args()


source, date, visibility_type, _ = get_source_date_type(args.data_file)
ms_path = f"./ms_data/{source}/{source}.u.{date}.{visibility_type}.ms"

if not os.path.exists(ms_path):
    importuvfits(args.data_file, vis=ms_path)
    print("Sucessfully transformed to ms format")
else:
    print("Already transformed to ms format")

