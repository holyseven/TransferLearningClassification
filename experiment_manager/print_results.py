import utils

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='run_classification', help='run_classification or run_segmentation')
FLAGS = parser.parse_args()

if __name__ == '__main__':
    logreader = utils.LogReader('../' + FLAGS.exp + '/log')
    logreader.print_necessary_logs(utils.list_toprint, ['network'])
