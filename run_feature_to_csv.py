import os
import argparse
from pathlib import Path
from interview_test import utils
from interview_test import features_preparing as fp


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-fp", "--save_file_path", help="Path where features should be stored", type=str,
                        default='')

    args = parser.parse_args()
    path_save_features = args.save_file_path
    if not Path(path_save_features).exists():
        os.makedirs(path_save_features)

    qry_string = "SELECT productid, brand, ram_gb, hdd_gb, ghz, price FROM test_table;"
    test_table_df = utils.sql_to_pandas(qry_string)

    fp.calculate_save_price_rank_info(test_table_df, path_save_features)
    fp.calculate_save_min_max_hdd_info(test_table_df, path_save_features)
    fp.calculate_save_median_ghz_by_ram_info(test_table_df, path_save_features)
