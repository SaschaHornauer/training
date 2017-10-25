import numpy as np
import time
import h5py
import torch
import torch.utils.data as data
import json
import sys
from random import shuffle
import os
import matplotlib.pyplot as plt
from random import shuffle
import random



class Dataset(data.Dataset):

    def __init__(self, data_folder_dir, require_one=[], ignore_list=[], stride=10, max_len=-1,
                 train_ratio=0.9, seed=None, nframes=2, nsteps=10, separate_frames=False,
                 metadata_shape=[], p_exclude_run=0., cache_file=None):
        self.max_len = max_len
        self.runs = os.walk(os.path.join(data_folder_dir, 'processed_h5py'), followlinks=True).next()[1]
        self.run_files = []

        # Initialize List of Files
        self.invisible = []
        self.visible = []
        self.total_length = 0 
        self.full_length = 0

        self.train_part = None
        self.val_part = None

        self.separate_frames = separate_frames

        self.train_ratio = train_ratio

        self.nframes = nframes
        self.nsteps = nsteps

        self.metadata_shape = metadata_shape

        random.seed(seed * 2)

        for run in self.runs:
            segs_in_run = os.walk(os.path.join(data_folder_dir, 'processed_h5py', run), followlinks=True).next()[1]

            run_labels = None
            try:
                run_labels = h5py.File(
                    os.path.join(data_folder_dir, 'processed_h5py', run, 'run_labels.h5py'),
                    'r')
            except Exception:
                continue

            if random.random() < p_exclude_run:
                continue

            # Ignore invalid runs
            ignored = False
            for ignore in ignore_list:
                if ignore in run_labels and run_labels[ignore][0]:
                    ignored = True
                    break
            if ignored:
                continue

            ignored = len(require_one) > 0 
            for require in require_one:
                if require in run_labels and run_labels[require][0]:
                    ignored = False
                    break
            if ignored:
                continue

            print 'Loading Run ' + run
            for seg in segs_in_run:
                images = h5py.File(
                    os.path.join(
                        data_folder_dir,
                        'processed_h5py',
                        run,
                        seg,
                        'images.h5py'),
                    'r')

                metadata = h5py.File(
                    os.path.join(data_folder_dir,
                        'processed_h5py',
                         run,
                         seg,
                         'metadata.h5py'),
                    'r')

                length = len(images['left'])

                self.run_files.append({'images': images, 'metadata': metadata, 'run_labels' : run_labels})
                self.visible.append(self.total_length)  # visible indicies

                # invisible is not actually used at all, but is extremely useful
                # for debugging indexing problems and gives very little slowdown
                self.invisible.append(self.full_length + 7) # actual indicies mapped

                self.total_length += (length - (self.nsteps * stride - 1) - 7)
                self.full_length += length

        # Create row gradient
        self.row_gradient = torch.FloatTensor(94, 168)
        for row in range(94):
            self.row_gradient[row, :] = row / 93.

        # Create col gradient
        self.col_gradient = torch.FloatTensor(94, 168)
        for col in range(168):
            self.col_gradient[:, col] = col / 167.

        self.stride = stride

        self.seed = seed or self.total_length
        self.subsampled_train_part = None

        self.cache_file = cache_file
        self.num_cache_points = None
        self.min_cache_points = None
        self.train_class_probs = None
        self.controls = {}

    def __getitem__(self, index):
        run_idx, t = self.create_map(index)

        data_file = self.run_files[run_idx]['images']
        metadata_file = self.run_files[run_idx]['metadata']

        list_camera_input = []

        for t in range(self.nframes):
            for camera in ('left', 'right'):
                list_camera_input.append(torch.from_numpy(data_file[camera][t]))
                camera_data = torch.cat(list_camera_input, 2)
                camera_data = camera_data.cuda().float() / 255. - 0.5
                camera_data = torch.transpose(camera_data, 0, 2)
                camera_data = torch.transpose(camera_data, 1, 2)

        # Get behavioral mode
        metadata_raw = self.run_files[run_idx]['run_labels']

        metadata = torch.FloatTensor(self.nframes, 64, 23, 41)
        metadata = torch.FloatTensor(*self.metadata_shape)

        metadata[:] = 0.
        for label_idx, cur_label in enumerate(['racing', 'follow', 'direct', 'play', 'furtive', 'clockwise', 'counterclockwise']):
            if self.separate_frames:
                metadata[:, label_idx, :, :] = int(cur_label in metadata_raw and metadata_raw[cur_label][0])
            else:
                metadata[label_idx, :, :] = int(cur_label in metadata_raw and metadata_raw[cur_label][0])

        # Get Ground Truth
        controls = []

        for i in range(0, self.stride * self.nsteps, self.stride):
            controls.append([float(self.run_files[run_idx]['metadata']['steer'][t + i]),
                             float(self.run_files[run_idx]['metadata']['motor'][t + i])])

        final_ground_truth = torch.FloatTensor(controls) / 99.

        return camera_data, metadata, final_ground_truth

    def __len__(self):
        if self.max_len == -1:
            return self.total_length
        return min(self.total_length, self.max_len)

    def train_len(self):
        return len(self.train_part)

    def val_len(self):
        return len(self.val_part)

    def get_train_partition(self):
        if self.train_part:
            return self.train_part
        else:
            self.train_part = set()
            self.val_part = set()
            random.seed(self.seed)
            for i in range(len(self)):
                if random.random() < self.train_ratio:
                    self.train_part.add(i)
                else:
                    self.val_part.add(i)
            return self.train_part

    def get_val_partition(self):
        if self.val_part:
            return self.val_part
        else:
            self.get_train_partition()
            return self.val_part

    def get_train_loader(self, p_subsample=None, seed=None, *args, **kwargs):
        random.seed(seed)
        remove_train, train_part = set(), set(self.train_part or self.get_train_partition())
        control_bins = [[0 for __ in range(0, 4)] for _ in range(0, 4)]

        if self.train_class_probs and self.controls:
            pass
        elif self.cache_file:
            try:
                js = json.load(open(self.cache_file, 'r'))
                self.train_class_probs, self.controls, self.num_cache_points, self.min_cache_points = js[0], js[1], js[2], js[3]
            except Exception as e:
                print(e)
                print(self.cache_file)
                print 'starting binning'
                _ = 0
                for i in train_part:
                    run_idx, t = self.create_map(i)
                    steer = int(float(self.run_files[run_idx]['metadata']['steer'][t]) / 25)
                    motor = int(float(self.run_files[run_idx]['metadata']['motor'][t]) / 25)
                    self.controls[str(i)] = (steer, motor)
                    control_bins[steer][motor] += 1
                    if _ % 10000 == 0:
                        print(str(_) + ' binned')
                    _ += 1
                self.num_cache_points = sum([sum(c) for c in control_bins])
                self.min_cache_points = min([min([c2 for c2 in c if c2 > 1000]) for c in control_bins if c > 1000])
                self.train_class_probs = [[self.min_cache_points / (c + 1e-32) for c in _] for _ in control_bins]
                print 'ending binning'
                json.dump([self.train_class_probs, self.controls, self.num_cache_points, self.min_cache_points], open(self.cache_file, 'w'))
        _ = 0
        for i in train_part:
            steer, motor = self.controls[str(i)][0], self.controls[str(i)][1]
            if random.random() > p_subsample * (self.num_cache_points / (8 * self.min_cache_points) * self.train_class_probs[steer][motor]):
                remove_train.add(i)
            if _ % 100000 == 0:
                print ('Trimming ' + str(_))
            _ += 1
        for i in remove_train:
            train_part.remove(i)

        self.subsampled_train_part = train_part

        kwargs['sampler'] = torch.utils.data.sampler.SubsetRandomSampler(list(train_part))
        return torch.utils.data.DataLoader(self, *args, **kwargs)

    def get_val_loader(self, *args, **kwargs):
        kwargs['sampler'] = torch.utils.data.sampler.SubsetRandomSampler(list(self.get_val_partition()))
        return torch.utils.data.DataLoader(self, *args, **kwargs)

    def create_map(self, global_index):
        for idx, length in enumerate(self.visible[::-1]):
            if global_index >= length:
                return len(self.visible) - idx - 1, global_index - length + 7

if __name__ == '__main__':
    train_dataset = Dataset('/hostroot/data/dataset/bair_car_data_Main_Dataset', ['furtive'], [])
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=500,
                                                    shuffle=False, pin_memory=False)
    start = time.time()
    for cam, meta, truth in train_data_loader:
        cur = time.time()
        print(500./(cur - start))
        start = cur
        pass
