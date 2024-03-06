import glob
import json
import os
from torch.autograd import Variable

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

class HParameters:
    def __init__(self):
        self.verbose = False
        self.use_cuda = True
        self.cuda_device = 0
        self.max_summary_length = 0.15

        self.epochs_max = 300
        self.stride_N = 3
        self.num_segments = 4

        self.datasets = ['data/datasets/eccv16_dataset_summe_google_pool5.h5',
                         'data/datasets/eccv16_dataset_tvsum_google_pool5.h5']

        self.splits = [ 'data/splits/summe_splits.json','data/splits/tvsum_splits.json']
        return


    def get_dataset_by_name(self, dataset_name):
        for d in self.datasets:
            if dataset_name in d:
                return [d]
        return None

    def load_from_args(self, args):
        for key in args:
            val = args[key]
            if val is not None:
                if hasattr(self, key) and isinstance(getattr(self, key), list):
                    val = val.split()
                setattr(self, key, val)


    def __str__(self):
        vars = [attr for attr in dir(self) if not callable(getattr(self,attr)) and not (attr.startswith("__") or attr.startswith("_"))]
        info_str = ''
        for i, var in enumerate(vars):
            val = getattr(self, var)
            if isinstance(val, Variable):
                val = val.data.cpu().numpy().tolist()[0]
            info_str += '['+str(i)+'] '+var+': '+str(val)+'\n'

        return info_str




