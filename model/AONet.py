import h5py
import random
import glob
import json
import os
from config import HParameters
from utils.vsum_tools import *
from utils.tools import *

def parse_splits_filename(splits_filename):
    # Parse split file and count number of k_folds
    spath, sfname = os.path.split(splits_filename)
    sfname, _ = os.path.splitext(sfname)
    dataset_name = sfname.split('_')[0]  # Get dataset name e.g. tvsum
    dataset_type = sfname.split('_')[1]  # augmentation type e.g. aug

    # The keyword 'splits' is used as the filename fields terminator from historical reasons.
    if dataset_type == 'splits':
        # Split type is not present
        dataset_type = ''

    # Get number of discrete splits within each split json file
    with open(splits_filename, 'r') as sf:
        splits = json.load(sf)

    return dataset_name, dataset_type, splits

def lookup_weights_splits_file(path, dataset_name, dataset_type, split_id):
    dataset_type_str = '' if dataset_type == '' else dataset_type + '_'
    weights_filename = path + '/models/{}_{}splits_{}_*.tar.pth'.format(dataset_name, dataset_type_str, split_id)
    weights_filename = glob.glob(weights_filename)
    if len(weights_filename) == 0:
        print("Couldn't find model weights: ", weights_filename)
        return ''

    # Get the first weights file in the dir
    weights_filename = weights_filename[0]
    splits_file = path + '/splits/{}_{}splits.json'.format(dataset_name, dataset_type_str)

    return weights_filename, splits_file

class AONet:
    def __init__(self, hps: HParameters):
        self.hps = hps
        self.model = None
        self.log_file = None
        self.verbose = hps.verbose

    def initialize(self, model, cuda_device=None):
        rnd_seed = 12345
        random.seed(rnd_seed)
        np.random.seed(rnd_seed)
        torch.manual_seed(rnd_seed)
        self.model = model

        cuda_device = cuda_device or self.hps.cuda_device

        if self.hps.use_cuda:
            print("Setting CUDA device: ", cuda_device)
            torch.cuda.set_device(cuda_device)
            torch.cuda.manual_seed(rnd_seed)

        if self.hps.use_cuda:
            self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        print("Initializing VASNet model and optimizer...")


    def load_datasets(self, datasets=None):
        """
        Loads all h5 datasets from the datasets list into a dictionary self.dataset
        referenced by their base filename
        :param datasets:  List of dataset filenames
        :return:
        """
        if datasets is None:
            datasets = self.hps.datasets

        datasets_dict = {}
        for dataset in datasets:
            _, base_filename = os.path.split(dataset)
            base_filename, _ = os.path.splitext(base_filename)
            print("Loading:", dataset)
            datasets_dict[base_filename] = h5py.File(dataset, 'r')

        self.datasets = datasets_dict
        return datasets_dict

    def load_split_file(self, splits_file):
        self.dataset_name, self.dataset_type, self.splits = parse_splits_filename(splits_file)
        n_folds = len(self.splits)
        self.split_file = splits_file
        print("Loading splits from: ", splits_file)

        return n_folds

    def select_split(self, split_id):
        print("Selecting split: ",split_id)

        self.split_id = split_id
        n_folds = len(self.splits)
        assert self.split_id < n_folds, "split_id (got {}) exceeds {}".format(self.split_id, n_folds)

        split = self.splits[self.split_id]
        self.train_keys = split['train_keys']
        self.test_keys = split['test_keys']

        dataset_filename = self.hps.get_dataset_by_name(self.dataset_name)[0]
        _,dataset_filename = os.path.split(dataset_filename)
        dataset_filename,_ = os.path.splitext(dataset_filename)


        self.train_keys = self.fix_keys(self.train_keys, dataset_filename)
        self.test_keys = self.fix_keys(self.test_keys, dataset_filename)
        return

    def fix_keys(self, keys, dataset_name=None):
        if len(self.datasets) == 1:
            dataset_name = next(iter(self.datasets))

        keys_out = []
        for key in keys:
            t = key.split('/')
            if len(t) != 2:
                assert dataset_name is not None, "ERROR dataset name in some keys is missing but there are multiple dataset {} to choose from".format(
                    len(self.datasets))

                key_name = dataset_name + '/' + key
                keys_out.append(key_name)
            else:
                keys_out.append(key)

        return keys_out

    def lookup_weights_file(self, data_path):
        dataset_type_str = '' if self.dataset_type == '' else self.dataset_type + '_'
        weights_filename = data_path + '/models/{}_splits_{}/G.pth.tar'.format(self.dataset_name,
                                                                                 self.split_id)
        weights_filename = glob.glob(weights_filename)
        if len(weights_filename) == 0:
            print("Couldn't find model weights: ", weights_filename)
            return ''

        # Get the first weights filename in the dir
        weights_filename = weights_filename[0]
        splits_file = data_path + '/splits/{}_{}splits.json'.format(self.dataset_name, dataset_type_str)

        return weights_filename, splits_file

    def load_model(self, model_filename):
        self.model.load_state_dict(torch.load(model_filename, map_location=lambda storage, loc: storage))
        return

    def get_data(self, key):
        key_parts = key.split('/')
        assert len(key_parts) == 2, "ERROR. Wrong key name: "+key
        dataset, key = key_parts
        return self.datasets[dataset][key]


    def eval_summary(self, machine_summary_activations, test_keys, results_filename=None, metric='tvsum', att_vecs=None):
        eval_metric = 'avg' if metric == 'tvsum' else 'max'

        if results_filename is not None:
            h5_res = h5py.File(results_filename, 'w')

        fms = []
        video_scores = []
        diversitys = []
        for key_idx, key in enumerate(test_keys):
            probs = machine_summary_activations[key]
            probs = norm_shot_score(probs)

            d = self.get_data(key)
            if 'change_points' not in d:
                print("ERROR: No change points in dataset/video ",key)

            cps = d['change_points'][...]
            num_frames = d['n_frames'][()]
            nfps = d['n_frame_per_seg'][...].tolist()
            positions = d['picks'][...]
            user_summary = d['user_summary'][...]
            seq = d['features'][...]
            machine_summary,picks_shot = select_shot(probs, cps, num_frames, nfps)

            fm, _, _ = evaluate_summary(machine_summary, user_summary, eval_metric)
            down_machine_summary = machine_summary[::15]
            diversity = self.get_summ_diversity(down_machine_summary, seq)
            diversitys.append(diversity)
            fms.append(fm)
            # Reporting & logging
            video_scores.append([key_idx + 1, key, "{:.1%}".format(fm)])

            if results_filename:
                gt = d['gtscore'][...]
                h5_res.create_dataset(key + '/score', data=probs)
                h5_res.create_dataset(key + '/machine_summary', data=machine_summary)
                h5_res.create_dataset(key + '/gtscore', data=gt)
                h5_res.create_dataset(key + '/fm', data=fm)
                h5_res.create_dataset(key + '/picks', data=positions)

                video_name = key.split('/')[1]
                if 'video_name' in d:
                    video_name = d['video_name'][...]
                h5_res.create_dataset(key + '/video_name', data=video_name)

                if att_vecs is not None:
                    h5_res.create_dataset(key + '/att', data=att_vecs[key])

        mean_fm = np.mean(fms)
        mean_diversity = np.mean(diversitys)
        # Reporting & logging
        if results_filename is not None:
            h5_res.close()

        return mean_fm, video_scores, mean_diversity


    def get_summ_diversity(self,pred_sum,features) -> float:
        """Evaluate diversity of the generated summary.

        :param pred_summ: Predicted down-sampled summary. Sized [N,].
        :param features: Normalized down-sampled video features. Sized [N, F].
        :return: Diversity value.
        """
        assert len(pred_sum) == len(features)
        pred_sum = np.asarray(pred_sum, dtype=bool)
        pos_features = features[pred_sum]

        if len(pos_features) < 2:
            return 0.0

        diversity = 0.0
        for feat in pos_features:
            diversity += (feat * pos_features).sum() - (feat * feat).sum()

        diversity /= len(pos_features) * (len(pos_features) - 1)
        return diversity