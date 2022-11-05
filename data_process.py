import torch
import torch.nn.parallel
import torch.utils.data
import numpy as np
import os
from tqdm import tqdm
import config
import suppor_lib
import pickle
import pickle as pck

'''
    Description: data_process.py is used to obtain Ground-truth scanpaths 
    for model training and evaluation. We give an example of how we process 
    Sitzmann database (https://github.com/vsitzmann/vr-saliency). One can use 
    this code for preparing his/her own training set.
    
    The data format of ``output.pkl`` is:
    [data]
        ['train']
            ['image1_name']
                ['image']: Tensor[3, 128, 256]
                ['scanpaths']: Tensor[n_scanpath, n_gaze_point, 3] # (x, y, z) for the 3-th dimension
            ['image2_name']
                ...
        ['test']
            ['imageN_name']
                ...
        ['info']
            ['train']: {
                'num_image': int,
                'num_scanpath': int,
                'scanpath_length': int,
                'max_scan_length': int
            }
            ['test']: {
                ...
            }
'''


def save_file(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
    f.close()


def load_logfile(path):
    log = pck.load(open(path, 'rb'), encoding='latin1')
    return log


def twoDict(pack, key_a, key_b, data):
    if key_a in pack:
        pack[key_a].update({key_b: data})
    else:
        pack.update({key_a: {key_b: data}})
    return pack


def create_info():
    info = {
        'train': {
            'num_image': 0,
            'num_scanpath': 0,
            'scanpath_length': 0,
            'max_scan_length': 0
        },
        'test': {
            'num_image': 0,
            'num_scanpath': 0,
            'scanpath_length': 0,
            'max_scan_length': 0
        }
    }
    return info


def summary(info):
    print("\n============================================")

    print("Train_set:   {} images, {} scanpaths,  length ={}".
          format(info['train']['num_image'], info['train']['num_scanpath'], info['train']['scanpath_length']))

    print("Test_set:    {} images, {} scanpaths,  length ={}".
          format(info['test']['num_image'], info['test']['num_scanpath'], info['test']['scanpath_length']))

    print("============================================\n")


def forward(database_name: str):
    if not os.path.exists('Datasets'):
        os.makedirs('Datasets')

    print('\nBegin process {} database'.format(database_name))

    if database_name == 'Sitzmann':
        data = Sitzmann_Dataset()
        dic = data.run()
        save_file('./Datasets/Sitzmann.pkl', dic)
        summary(dic['info'])

    else:
        print('\nYou need to prepare the code for {} data processing'.format(database_name))


class Sitzmann_Dataset():
    def __init__(self):
        super().__init__()
        self.images_path = config.dic_Sitzmann['IMG_PATH']
        self.gaze_path = config.dic_Sitzmann['GAZE_PATH']
        self.test_set = config.dic_Sitzmann['TEST_SET']
        self.duration = 30
        self.info = create_info()
        self.images_test_list = []
        self.images_train_list = []
        self.image_and_scanpath_dict = {}

    def mod(self, a, b):
        c = a // b
        r = a - c * b
        return r

    def rotate(self, lat_lon, angle):
        # We convert [-180, 180] to [0, 360], then compute the new longitude.
        # We ``minus`` the angle here, which is different from what we do in rotating images,
        # because ffmepeg has a different coordination. For example, set ``yaw=60`` in ffmepg
        # equate to longitude = -60 + longitude
        new_lon = self.mod(lat_lon[:, 1] + 180 - angle, 360) - 180
        rotate_lat_lon = lat_lon
        rotate_lat_lon[:, 1] = new_lon
        return rotate_lat_lon

    def handle_empty(self, sphere_coords):
        empty_index = np.where(sphere_coords[:, 0] == -999)[0]
        throw = False
        for _index in range(empty_index.shape[0]):
            # if not throw the scanpath of this user
            if not throw:
                # if the first one second is empty
                if empty_index[_index] == 0:
                    # if the next second is not empty
                    if sphere_coords[empty_index[_index] + 1, 0] != -999:
                        sphere_coords[empty_index[_index], 0] = sphere_coords[empty_index[_index] + 1, 0]
                        sphere_coords[empty_index[_index], 1] = sphere_coords[empty_index[_index] + 1, 1]
                    else:
                        throw = True
                        # print(" Too many invalid gaze points !! {}".format(empty_index))

                # if the last one second is empty
                elif empty_index[_index] == (self.duration - 1):
                    sphere_coords[empty_index[_index], 0] = sphere_coords[empty_index[_index] - 1, 0]
                    sphere_coords[empty_index[_index], 1] = sphere_coords[empty_index[_index] - 1, 1]

                else:
                    prev_x = sphere_coords[empty_index[_index] - 1, 1]
                    prev_y = sphere_coords[empty_index[_index] - 1, 0]
                    next_x = sphere_coords[empty_index[_index] + 1, 1]
                    next_y = sphere_coords[empty_index[_index] + 1, 0]

                    if prev_x == -999 or next_x == -999:
                        throw = True
                        # print(" Too many invalid gaze points !! {}".format(empty_index))

                    else:
                        " Interpolate on lat "
                        sphere_coords[empty_index[_index], 0] = 0.5 * (prev_y + next_y)

                        " Interpolate on lon "
                        # the maximum distance between two points on a sphere is pi
                        if np.abs(next_x - prev_x) <= 180:
                            sphere_coords[empty_index[_index], 1] = 0.5 * (prev_x + next_x)
                        # jump to another side
                        else:
                            true_distance = 360 - np.abs(next_x - prev_x)
                            if next_x > prev_x:
                                _temp = prev_x - true_distance / 2
                                if _temp < -180:
                                    _temp = 360 + _temp
                            else:
                                _temp = prev_x + true_distance / 2
                                if _temp > 180:
                                    _temp = _temp - 360
                            sphere_coords[empty_index[_index], 1] = _temp

        return sphere_coords, throw

    def sample_gaze_points(self, raw_data):
        fixation_coords = []
        samples_per_bin = raw_data.shape[0] // self.duration
        bins = raw_data[:samples_per_bin * self.duration].reshape([self.duration, -1, 2])
        for bin in range(self.duration):
            " filter out invalid gaze points "
            _fixation_coords = bins[bin, np.where((bins[bin, :, 0] != 0) & (bins[bin, :, 1] != 0))]
            if _fixation_coords.shape[1] == 0:
                " mark the empty set"
                fixation_coords.append([-999, -999])
            else:
                " sample the first element in a set of one-second gaze points "
                sample_vale = _fixation_coords[0, 0]
                fixation_coords.append(sample_vale)
        sphere_coords = np.vstack(fixation_coords) - [90, 180]

        return sphere_coords

    def get_train_set(self):

        all_files = [os.path.join(self.gaze_path, self.images_train_list[i].split('/')[-1].split('.')[0][:-2] + '.pck')
                     for i in range(0, len(self.images_train_list), 6)]

        runs_files = [load_logfile(logfile) for logfile in all_files]

        image_id = 0

        original_image_id = 0

        for run in runs_files:
            temple_gaze = np.zeros((len(run['data']), 30, 2))
            scanpath_id = 0

            " save scanpath data to ``temple_gaze`` "
            for data in run['data']:
                relevant_fixations = data['gaze_lat_lon']

                if len(relevant_fixations.shape) > 1:
                    norm_coords = self.sample_gaze_points(relevant_fixations)
                else:
                    continue

                " handle invalid set"
                norm_coords, throw = self.handle_empty(norm_coords)

                if throw:  # throw this scanpath if too many invalid values.
                    continue
                else:
                    norm_coords = torch.from_numpy(norm_coords.copy())
                    temple_gaze[scanpath_id] = norm_coords
                    scanpath_id += 1

            temple_gaze = temple_gaze[:scanpath_id]

            original_image_id += 1

            " rotate scanpaths "
            for rotation_id in range(6):

                image = suppor_lib.image_process(self.images_train_list[image_id])
                gaze_ = np.zeros((temple_gaze.shape[0], 30, 3))
                rotation_angle = rotation_id * 60 - 180

                for scanpath_id in range(0, temple_gaze.shape[0]):
                    gaze_[scanpath_id] = suppor_lib.sphere2xyz(
                        torch.from_numpy(self.rotate(temple_gaze[scanpath_id], rotation_angle)))

                    self.info['train']['num_scanpath'] += 1

                self.info['train']['num_image'] += 1

                dic = {"image": image, "scanpaths": gaze_}

                twoDict(self.image_and_scanpath_dict, "train",
                        self.images_train_list[image_id].split('/')[-1].split('.')[0],
                        dic)

                print('Processing - {}. [Filter out {} scanpaths]'
                      .format(self.images_train_list[image_id].split('/')[-1],
                              len(run['data']) - scanpath_id - 1))

                image_id += 1

        self.info['train']['scanpath_length'], self.info['test']['scanpath_length'] = self.duration, self.duration

    def get_test_set(self):

        all_files = [os.path.join(self.gaze_path, self.images_test_list[i].split('/')[-1].split('.')[0] + '.pck')
                     for i in range(len(self.images_test_list))]

        runs_files = [load_logfile(logfile) for logfile in all_files]

        image_id = 0

        for run in runs_files:

            scanpath_id = 0
            gaze_ = np.zeros((len(run['data']), 30, 3))
            image = suppor_lib.image_process(self.images_test_list[image_id])

            for data in run['data']:

                relevant_fixations = data['gaze_lat_lon']

                if len(relevant_fixations.shape) > 1:
                    sphere_coords = self.sample_gaze_points(relevant_fixations)
                else:
                    continue

                " handle invalid set"
                sphere_coords, throw = self.handle_empty(sphere_coords)

                if throw:  # throw this scanpath if too many invalid values.
                    continue
                else:
                    sphere_coords = torch.from_numpy(sphere_coords.copy())
                    gaze_[scanpath_id] = suppor_lib.sphere2xyz(sphere_coords)
                    scanpath_id += 1
                    self.info['test']['num_scanpath'] += 1

            gaze = gaze_[:scanpath_id]

            dic = {"image": image, "scanpaths": gaze}

            twoDict(self.image_and_scanpath_dict, "test",
                    self.images_test_list[image_id].split('/')[-1].split('.')[0],
                    dic)

            print('Processing - {}. [Filter out {} scanpaths]'
                  .format(self.images_test_list[image_id].split('/')[-1], gaze_.shape[0] - scanpath_id))

            image_id += 1

        self.info['test']['num_image'] = image_id

    def run(self):
        ''
        ' @@ PATH PREPARE '
        for file_name in os.listdir(self.images_path):
            if ".png" in file_name:
                if file_name in self.test_set:
                    self.images_test_list.append(os.path.join(self.images_path, file_name))
                else:
                    self.images_train_list.append(os.path.join(self.images_path, file_name))

        ' @@ GET TRAINING SET '
        print('\nProcessing [Training Set]\n')
        self.get_train_set()

        ' @@ GET TEST SET '
        print('\nProcessing [Test Set]\n')
        self.get_test_set()

        ' @@ RECORD DATABASE INFORMATION '
        self.image_and_scanpath_dict['info'] = self.info

        return self.image_and_scanpath_dict




if __name__ == '__main__':

    # 1. Rotate 360-degree images for data augmentation using:
    # suppor_lib.rotate_images(input_path, output_path)

    # 2. Modify the configure in config.py

    # 3. Prepare the dataset.
    Datasets = ['Sitzmann']
    for dataset in Datasets:
        forward(dataset)
