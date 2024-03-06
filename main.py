import argparse
import sys
import time
import os
import numpy as np
from config import HParameters
from model.VSNet import VSNet
from model.AONet import parse_splits_filename
from utils.tools import print_table


def eval_split(hps, splits_filename, data_dir='data'):
    print("\n")
    ao = VSNet(hps)
    ao.load_datasets()
    ao.load_split_file(splits_filename)

    val_fscores = []
    for split_id in range(len(ao.splits)):
        ao.select_split(split_id)
        weights_filename, _ = ao.lookup_weights_file(data_dir)
        print("Loading model:", weights_filename)
        ao.load_model(weights_filename)
        val_fscore, video_scores,diversity_score = ao.eval(ao.test_keys)
        val_fscores.append(val_fscore)
        val_fscore_avg = np.mean(val_fscores)

        if hps.verbose:
            video_scores = [["No.", "Video", "F-score"]] + video_scores
            print_table(video_scores, cell_width=[4,45,5])

        print("Avg F-score: ", val_fscore)
        print("")

    print("Total AVG F-score: ", val_fscore_avg)
    return val_fscore_avg


def train(hps):
    st = time.perf_counter()
    os.makedirs(hps.output_dir, exist_ok=True)
    os.makedirs(os.path.join(hps.output_dir, 'splits'), exist_ok=True)
    os.makedirs(os.path.join(hps.output_dir, 'models'), exist_ok=True)
    # os.system('cp -f data/splits/*.json  ' + hps.output_dir + '/splits/')

    # Create a file to collect results from all splits
    f = open(hps.output_dir + '/results.txt', 'wt')

    for split_filename in hps.splits:
        dataset_name, dataset_type, splits = parse_splits_filename(split_filename)

        datasets = None
        if dataset_type == '':
            datasets = hps.get_dataset_by_name(dataset_name)

        if datasets is None:
            datasets = hps.datasets

        f_avg = 0
        max_fscore_diversity_avg = 0
        n_folds = len(splits)
        for split_id in range(n_folds):
            ao = VSNet(hps)
            ao.load_datasets(datasets=datasets)
            ao.load_split_file(splits_file=split_filename)
            ao.select_split(split_id=split_id)
            fscore, fscore_epoch, max_fscore_diversity = ao.train()
            f_avg += fscore
            max_fscore_diversity_avg += max_fscore_diversity
            # Log F-score for this split_id
            f.write(split_filename + ', ' + str(split_id) + ', ' + str(fscore) + ', ' + str(max_fscore_diversity_avg) + ', ' + str(fscore_epoch) + '\n')
            f.flush()

            # Save model with the highest F score
            _, log_file = os.path.split(split_filename)
            log_dir, _ = os.path.splitext(log_file)
            log_dir += '_' + str(split_id)
            log_file = os.path.join(hps.output_dir, 'models', log_dir) + '_' + str(fscore) + '.tar.pth'

            os.makedirs(os.path.join(hps.output_dir, 'models', ), exist_ok=True)


            print("Split: {0:}   Best F-score: {1:0.5f}   Model: {2:}".format(split_filename, fscore, log_file))

        # Write average F-score for all splits to the results.txt file
        f_avg /= n_folds
        f.write(split_filename + ', ' + str('avg') + ', ' + str(f_avg) + '\n')
        f.flush()
    end = time.perf_counter()
    t = end - st
    f.write('total time:{:.3f} h\n'.format(t / 3600))
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PyTorch implementation of H-SMAE")
    parser.add_argument('-r', '--root', type=str, default='', help="Project root directory")
    parser.add_argument('-d', '--datasets', type=str, help="Path to a comma separated list of h5 datasets")
    parser.add_argument('-s', '--splits', type=str, help="Comma separated list of split files.")
    parser.add_argument('-t', '--train', default= True , action='store_true', help="Train")
    parser.add_argument('-v', '--verbose', action='store_true', help="Prints out more messages")
    parser.add_argument('-o', '--output-dir', type=str, default='data', help="Experiment name")

    args = parser.parse_args()

    # MAIN
    #======================
    hps = HParameters()
    hps.load_from_args(args.__dict__)

    print("Parameters:")
    print("----------------------------------------------------------------------")
    print(hps)


    if hps.train:
        train(hps)
    else:
        results=[['No', 'Split', 'Mean F-score']]
        for i, split_filename in enumerate(hps.splits):
            f_score = eval_split(hps, split_filename, data_dir=hps.output_dir)
            results.append([i+1, split_filename, str(round(f_score * 100.0, 3))+"%"])

        print("\nFinal Results:")
        print_table(results)

    sys.exit(0)

