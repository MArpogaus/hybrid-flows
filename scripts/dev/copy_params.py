import argparse
import logging
from argparse import FileType

import yaml

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def split_keys(keys_str):
    return keys_str.split(".")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Copy Parameters from on yaml to another"
    )
    parser.add_argument(
        "source_file",
        help="Source Parameter File",
        type=FileType("r"),
    )
    parser.add_argument(
        "keys",
        help="Key to copy to Destination Parameter File",
        type=split_keys,
    )
    parser.add_argument(
        "dest_files",
        help="Destination Parameter File",
        type=FileType("r+"),
        nargs="+",
    )
    return parser.parse_args()


def get_target_value_from_dict(params_dict, keys):
    for key in keys[:-1]:
        if key not in params_dict.keys():
            params_dict[key] = {}
        params_dict = params_dict[key]

    return params_dict, keys[-1]


def update_params_dict(source_params, dest_params, keys):
    dest_params, key = get_target_value_from_dict(dest_params, keys)
    dest_params[key] = source_params


def main(source_file, dest_files, keys):
    with source_file as sf:
        LOGGER.info(f"reading file: {sf.name}")
        source_yaml = yaml.safe_load(sf)
    source_params, key = get_target_value_from_dict(source_yaml, keys)
    source_params = source_params[key]
    for dest_file in dest_files:
        with dest_file as df:
            LOGGER.info(f"updating file: {df.name}")
            dest_yaml = yaml.safe_load(df)
            update_params_dict(source_params, dest_yaml, keys)
            df.seek(0)
            yaml.safe_dump(dest_yaml, df)
            df.truncate()


if __name__ == "__main__":
    args = parse_arguments()
    main(args.source_file, args.dest_files, args.keys)
