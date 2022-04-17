import os
import argparse
from pathlib import Path
from interview_test import modeling as ml


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", "--data_file_path", help="Path where data is stored", type=str, default='')
    parser.add_argument("-mlp", "--model_path", help="Path where ml models will be stored", type=str,
                        default=f'{os.getcwd()}/model')

    args = parser.parse_args()
    data_path = args.data_file_path
    model_path = args.model_path
    if not Path(model_path).exists():
        os.makedirs(model_path)

    ml.train_model_pipeline(data_path, model_path)
