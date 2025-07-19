# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import pathlib
import warnings

from logging import getLogger

import numpy as np
import pandas as pd

from decord import VideoReader, cpu

import torch

from src.datasets.utils.weighted_sampler import DistributedWeightedSampler

_GLOBAL_SEED = 0
logger = getLogger()


def make_videodataset(
    data_paths,
    batch_size,
    frames_per_clip=8,
    frame_step=4,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(10**9),
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    duration=None,
    log_dir=None,
):
    dataset = VideoDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        duration=duration,
        shared_transform=shared_transform,
        transform=transform)

    logger.info('VideoDataset dataset created')
    if datasets_weights is not None:
        dist_sampler = DistributedWeightedSampler(
            dataset.sample_weights,
            num_replicas=world_size,
            rank=rank,
            shuffle=True)
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=num_workers > 0)
    logger.info('VideoDataset unsupervised data loader created')

    return dataset, data_loader, dist_sampler


class VideoDataset(torch.utils.data.Dataset):
    """ Video classification dataset. """

    def __init__(
        self,
        data_paths,
        datasets_weights=None,
        frames_per_clip=16,
        frame_step=4,
        num_clips=1,
        transform=None,
        shared_transform=None,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
        duration=None,  # duration in seconds
    ):
        self.data_paths = data_paths
        self.datasets_weights = datasets_weights
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.duration = duration

        if VideoReader is None:
            raise ImportError('Unable to import "decord" which is required to read videos.')

        # Load video paths and labels
        samples, labels = [], []
        self.num_samples_per_dataset = []
        for data_path in self.data_paths:

            if data_path[-4:] == '.csv':
                data = pd.read_csv(data_path, header=None, delimiter=" ")
                samples += list(data.values[:, 0])
                labels += list(data.values[:, 1])
                num_samples = len(data)
                self.num_samples_per_dataset.append(num_samples)

            elif data_path[-4:] == '.npy':
                data = np.load(data_path, allow_pickle=True)
                data = list(map(lambda x: repr(x)[1:-1], data))
                samples += data
                labels += [0] * len(data)
                num_samples = len(data)
                self.num_samples_per_dataset.append(len(data))

        # [Optional] Weights for each sample to be used by downstream
        # weighted video sampler
        self.sample_weights = None
        if self.datasets_weights is not None:
            self.sample_weights = []
            for dw, ns in zip(self.datasets_weights, self.num_samples_per_dataset):
                self.sample_weights += [dw / ns] * ns

        self.samples = samples
        self.labels = labels

    def __getitem__(self, index):
        sample = self.samples[index]

        # Keep trying to load videos until you find a valid sample
        loaded_video = False
        while not loaded_video:
            buffer, clip_indices = self.loadvideo_decord(sample)  # [T H W 3]
            loaded_video = len(buffer) > 0
            if not loaded_video:
                index = np.random.randint(self.__len__())
                sample = self.samples[index]

        # Label/annotations for video
        label = self.labels[index]

        def split_into_clips(video):
            """ Split video into a list of clips """
            fpc = self.frames_per_clip
            nc = self.num_clips
            return [video[i*fpc:(i+1)*fpc] for i in range(nc)]

        # Parse video into frames & apply data augmentations
        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
        buffer = split_into_clips(buffer)
        if self.transform is not None:
            buffer = [self.transform(clip) for clip in buffer]

        return buffer, label, clip_indices

    def loadvideo_decord(self, sample):
        """ Load video content using Decord """

        fname = sample
        if not os.path.exists(fname):
            warnings.warn(f'video path not found {fname=}')
            return [], None

        _fsize = os.path.getsize(fname)
        if _fsize < 1 * 1024:  # avoid hanging issue
            warnings.warn(f'video too short {fname=}')
            return [], None
        if _fsize > self.filter_long_videos:
            warnings.warn(f'skipping long video of size {_fsize=} (bytes)')
            return [], None

        try:
            vr = VideoReader(fname, num_threads=-1, ctx=cpu(0))
        except Exception:
            return [], None

        fpc = self.frames_per_clip
        fstp = self.frame_step
        if self.duration is not None:
            try:
                fps = vr.get_avg_fps()
                fstp = int(self.duration * fps / fpc)
            except Exception as e:
                warnings.warn(e)
        clip_len = int(fpc * fstp)

        if self.filter_short_videos and len(vr) < clip_len:
            warnings.warn(f'skipping video of length {len(vr)}')
            return [], None

        vr.seek(0)  # Go to start of video before sampling frames

        # Partition video into equal sized segments and sample each clip
        # from a different segment
        partition_len = len(vr) // self.num_clips

        all_indices, clip_indices = [], []
        for i in range(self.num_clips):

            if partition_len > clip_len:
                # If partition_len > clip len, then sample a random window of
                # clip_len frames within the segment
                end_indx = clip_len
                if self.random_clip_sampling:
                    end_indx = np.random.randint(clip_len, partition_len)
                start_indx = end_indx - clip_len
                indices = np.linspace(start_indx, end_indx, num=fpc)
                indices = np.clip(indices, start_indx, end_indx-1).astype(np.int64)
                # --
                indices = indices + i * partition_len
            else:
                # If partition overlap not allowed and partition_len < clip_len
                # then repeatedly append the last frame in the segment until
                # we reach the desired clip length
                if not self.allow_clip_overlap:
                    indices = np.linspace(0, partition_len, num=partition_len // fstp)
                    indices = np.concatenate((indices, np.ones(fpc - partition_len // fstp) * partition_len,))
                    indices = np.clip(indices, 0, partition_len-1).astype(np.int64)
                    # --
                    indices = indices + i * partition_len

                # If partition overlap is allowed and partition_len < clip_len
                # then start_indx of segment i+1 will lie within segment i
                else:
                    sample_len = min(clip_len, len(vr)) - 1
                    indices = np.linspace(0, sample_len, num=sample_len // fstp)
                    indices = np.concatenate((indices, np.ones(fpc - sample_len // fstp) * sample_len,))
                    indices = np.clip(indices, 0, sample_len-1).astype(np.int64)
                    # --
                    clip_step = 0
                    if len(vr) > clip_len:
                        clip_step = (len(vr) - clip_len) // (self.num_clips - 1)
                    indices = indices + i * clip_step

            clip_indices.append(indices)
            all_indices.extend(list(indices))

        buffer = vr.get_batch(all_indices).asnumpy()
        return buffer, clip_indices

    def __len__(self):
        return len(self.samples)

if __name__ == '__main__':
    import sys
    import os
    
    # Add the project root to Python path so we can import src modules
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, project_root)
    
    import yaml
    
    # Load the config file
    config_path = 'configs/pretrain/vit_tiny.yaml'
    with open(config_path, 'r') as y_file:
        config = yaml.load(y_file, Loader=yaml.FullLoader)
    
    print("Loaded config:")
    print(f"Dataset paths: {config['data']['datasets']}")
    print(f"Batch size: {config['data']['batch_size']}")
    print(f"Num frames: {config['data']['num_frames']}")
    print(f"Num clips: {config['data']['num_clips']}")
    print(f"Sampling rate: {config['data']['sampling_rate']}")
    
    # Extract parameters from config
    data_config = config['data']
    model_config = config['model']
    
    # Import mask collator
    from src.masks.multiblock3d import MaskCollator as MB3DMaskCollator
    from src.masks.random_tube import MaskCollator as TubeMaskCollator
    
    # Create mask collator based on config
    mask_type = model_config.get('mask_type', 'multiblock3d')
    cfgs_mask = [{'aspect_ratio': [0.75, 1.5], 'num_blocks': 8, 'spatial_scale': [0.15, 0.15], 'temporal_scale': [1.0, 1.0], 'max_temporal_keep': 1.0, 'max_keep': None}, {'aspect_ratio': [0.75, 1.5], 'num_blocks': 2, 'spatial_scale': [0.7, 0.7], 'temporal_scale': [1.0, 1.0], 'max_temporal_keep': 1.0, 'max_keep': None}, {'mode': 'time_split'}]

    if mask_type == 'multiblock3d':
        print('Initializing basic multi-block mask')
        mask_collator = MB3DMaskCollator(
            crop_size=model_config.get('crop_size', 224),
            num_frames=data_config['num_frames'],
            patch_size=model_config.get('patch_size', 16),
            tubelet_size=model_config.get('tubelet_size', 2),
            cfgs_mask=cfgs_mask)
    else:
        print('Initializing random tube mask')
        mask_collator = TubeMaskCollator(
            crop_size=model_config.get('crop_size', 224),
            num_frames=data_config['num_frames'],
            patch_size=model_config.get('patch_size', 16),
            tubelet_size=model_config.get('tubelet_size', 2),
            cfgs_mask=cfgs_mask)
    
    print('mask_collator', type(mask_collator))
    
    # Run the make_videodataset function
    try:
        dataset, data_loader, dist_sampler = make_videodataset(
            data_paths=data_config['datasets'],
            batch_size=data_config['batch_size'],
            frames_per_clip=data_config['num_frames'],
            frame_step=data_config['sampling_rate'],
            num_clips=data_config['num_clips'],
            random_clip_sampling=True,
            allow_clip_overlap=False,
            filter_short_videos=data_config.get('filter_short_videos', False),
            filter_long_videos=int(10**9),
            transform=None,
            shared_transform=None,
            rank=0,
            world_size=1,
            datasets_weights=None,
            collator=mask_collator,
            drop_last=True,
            num_workers=data_config.get('num_workers', 4),
            pin_mem=data_config.get('pin_mem', True),
            duration=data_config.get('clip_duration', None),
            log_dir=None,
        )
        
        print(f"\nSuccessfully created dataset!")
        print(f"Dataset length: {len(dataset)}")
        print(f"DataLoader length: {len(data_loader)}")
        
        # Test loading a batch
        print("\nTesting data loading...")
        loader = iter(data_loader)
        
        # Explore the content of the loader - match training script format
        print('=== DEBUG: Printing loader content ===')
        try:
            # Get first few items from loader to inspect structure
            for i in range(3):  # Print first 3 items
                try:
                    item = next(loader)
                    print(f'Loader item {i}:')
                    print(f'  Type: {type(item)}')
                    print(f'  Length: {len(item)}')
                    if len(item) >= 3:
                        udata, masks_enc, masks_pred = item
                        print(f'  udata type: {type(udata)}')
                        print(f'  udata length: {len(udata)}')
                        if len(udata) > 0:
                            print(f'  udata[0] type: {type(udata[0])}')
                            print(f'  udata[0] length: {len(udata[0])}')
                            if len(udata[0]) > 0:
                                print(f'  udata[0][0] type: {type(udata[0][0])}')
                                print(f'  udata[0][0] shape: {udata[0][0].shape}')
                        print(f'  masks_enc type: {type(masks_enc)}')
                        print(f'  masks_enc length: {len(masks_enc)}')
                        for i, m in enumerate(masks_enc):
                            print(f'  masks_enc[{i}] type: {type(m)}')
                            print(f'  masks_enc[{i}] shape: {m.shape}')
                        print(f'  masks_pred type: {type(masks_pred)}')
                        print(f'  masks_pred length: {len(masks_pred)}')
                        for i, m in enumerate(masks_pred):
                            print(f'  masks_pred[{i}] type: {type(m)}')
                            print(f'  masks_pred[{i}] shape: {m.shape}')
                    print('  ---')
                except StopIteration:
                    print(f'Loader exhausted after {i} items')
                    break
        except Exception as e:
            print(f'Error inspecting loader: {e}')
        
        print('=== END DEBUG ===')
            
    except Exception as e:
        print(f"Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
