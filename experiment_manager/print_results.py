import utils

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='run_classification', help='run_classification or run_segmentation')
FLAGS = parser.parse_args()

if __name__ == '__main__':
    logreader = utils.LogReader('../' + FLAGS.exp + '/log')
    filter_dict = {'database': 'caltech256'}
    # filter_dict = {'weight_decay_mode': 1}
    logreader.print_necessary_logs(utils.list_toprint, filter_dict)
