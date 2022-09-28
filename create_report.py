#!/usr/bin/env python2
##########################################################
# Copyright 2019 Oath Inc.
# Licensed under the terms of the MIT license.
# Please see LICENSE file in the project root for terms.
##########################################################
#
# Usage examples
# ==============
#
# Print the help message:
# $ ./create_report.py --help
#
# Create a report based on a single run:
# $ ./create_report.py  --pickle eval_results/fc_transformer_bu_adaptive_test_report_data.pkl  --out_dir reports/single_run/
#
# Create the report containing results from two runs:
# $ ./create_report.py  --pickle transformer_results/fc_transformer_bu_adaptive_test_report_data.pkl relation_transformer_results/fc_transformer_bu_adaptive_test_report_data.pkl  --run_names transformer relation_transformer  --out_dir reports/two_runs/
#
# After creating the report by using the commands above, you can set up a
# server to view it using something like the following commands:
# $ cd reports/
# $ python -m SimpleHTTPServer 8888

import argparse
from misc.report import create_report, ReportConfig, ReportData
from datetime import datetime


class Args:
    BASE_OUT_DIR = 'out_dir'
    REPORT_DATA_PICKLES = 'pickles'
    RUN_NAMES = 'run_names'
    ADD_TIME = 'add_time'
    NO_ADD_TIME = 'no_add_time'


def main():
    args = _get_command_line_arguments()
    base_out_dir = args[Args.BASE_OUT_DIR]
    add_time = args[Args.ADD_TIME]
    out_dir = _get_out_dir(base_out_dir, add_time)
    pickle_paths = args[Args.REPORT_DATA_PICKLES]
    run_names = args[Args.RUN_NAMES]
    report_data_list = [
        ReportData.read_from_pickle(pickle_path, run_name) for
        pickle_path, run_name in zip(pickle_paths, run_names)]
    create_report(report_data_list, ReportConfig(out_dir))


def _get_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--' + Args.REPORT_DATA_PICKLES,
        help='Pickle files with the ReportData objects', required=True,
        nargs='+')
    parser.add_argument(
        '--' + Args.RUN_NAMES, help='A name for each run', required=True,
        nargs='+')
    parser.add_argument(
        '--' + Args.BASE_OUT_DIR, help='Output directory', required=True)
    parser.add_argument(
        '--' + Args.ADD_TIME, help='Add a timestamp to the output directory',
        default=False, required=False, action='store_true', dest=Args.ADD_TIME)
    parser.add_argument(
        '--' + Args.NO_ADD_TIME,
        help='Don\'t add a timestamp to the output directory',
        required=False, action='store_false', dest=Args.ADD_TIME)
    args_dict = vars(parser.parse_args())
    return args_dict


def _get_out_dir(base_out_dir, add_time):
    if add_time:
        date_string = datetime.now().strftime('%Y-%m-%d--%H_%M_%S')
        out_dir = "%s_%s" % (base_out_dir, date_string)
    else:
        out_dir = base_out_dir
    return out_dir


if __name__ == '__main__':
    main()
