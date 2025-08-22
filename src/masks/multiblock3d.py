# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math

from multiprocessing import Value

from logging import getLogger

import torch

_GLOBAL_SEED = 0
logger = getLogger()


class MaskCollator(object):

    def __init__(
        self,
        cfgs_mask,
        crop_size=(224, 224),
        num_frames=16,
        patch_size=(16, 16),
        tubelet_size=2,
    ):
        super(MaskCollator, self).__init__()

        self.mask_generators = []
        for m in cfgs_mask:
            mask_generator = _MaskGenerator(
                crop_size=crop_size,
                num_frames=num_frames,
                spatial_patch_size=patch_size,
                temporal_patch_size=tubelet_size,
                spatial_pred_mask_scale=m.get('spatial_scale'),
                temporal_pred_mask_scale=m.get('temporal_scale'),
                aspect_ratio=m.get('aspect_ratio'),
                npred=m.get('num_blocks'),
                max_context_frames_ratio=m.get('max_temporal_keep', 1.0),
                max_keep=m.get('max_keep', None),
                mode=m.get('mode', 'block'),
            )
            self.mask_generators.append(mask_generator)

    def step(self):
        for mask_generator in self.mask_generators:
            mask_generator.step()
            
    def __call__(self, batch):
        print('Inside MaskCollator')
        print(f"Batch length: {len(batch)}")
        print(f"Batch first element first element type: {type(batch[0][0])}")
        print(f"Batch first element first element len: {len(batch[0][0])}")
         
        collated_batch = torch.utils.data.default_collate(batch)
        print(f"Collated batch type: {type(collated_batch)}")
        print(f"Collated batch length: {len(collated_batch)}")
        print(f"Collated batch type: {type(collated_batch[0])}")
        print(f"Collated batch length: {len(collated_batch[0])}")
        print(f"Collated batch type: {type(collated_batch[0][0])}")
        
        collated_masks_pred, collated_masks_enc = [], []
        for i, mask_generator in enumerate(self.mask_generators):
            masks_enc, masks_pred = mask_generator(collated_batch[0][0])
            collated_masks_enc.append(masks_enc)
            collated_masks_pred.append(masks_pred)
        
 
        print(f"Collated masks enc type: {collated_masks_enc[0].shape}")
        print(f"Collated masks enc len: {len(collated_masks_enc)}")
        print(f"Collated masks pred shape: {collated_masks_pred[0].shape}")
        print(f"Collated masks pred shape: {collated_masks_pred[1].shape}")
        print(f"Collated masks pred len: {len(collated_masks_pred)}")
        return collated_batch, collated_masks_enc, collated_masks_pred


class _MaskGenerator(object):

    def __init__(
        self,
        crop_size=(224, 224),
        num_frames=16,
        spatial_patch_size=(16, 16),
        temporal_patch_size=2,
        spatial_pred_mask_scale=(0.2, 0.8),
        temporal_pred_mask_scale=(1.0, 1.0),
        aspect_ratio=(0.3, 3.0),
        npred=1,
        max_context_frames_ratio=1.0,
        max_keep=None,
        mode='block',
    ):
        super(_MaskGenerator, self).__init__()
        if not isinstance(crop_size, tuple):
            crop_size = (crop_size, ) * 2
        self.crop_size = crop_size
        self.height, self.width = crop_size[0] // spatial_patch_size, crop_size[1] // spatial_patch_size
        self.duration = num_frames // temporal_patch_size

        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.aspect_ratio = aspect_ratio
        self.spatial_pred_mask_scale = spatial_pred_mask_scale
        self.temporal_pred_mask_scale = temporal_pred_mask_scale
        self.npred = npred
        self.max_context_duration = max(1, int(self.duration * max_context_frames_ratio))  # maximum number of time-steps (frames) spanned by context mask
        self.max_keep = max_keep  # maximum number of patches to keep in context
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes
        self.mode = mode
    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(
        self,
        generator,
        temporal_scale,
        spatial_scale,
        aspect_ratio_scale
    ):
        # -- Sample temporal block mask scale
        _rand = torch.rand(1, generator=generator).item()
        min_t, max_t = temporal_scale
        temporal_mask_scale = min_t + _rand * (max_t - min_t)
        t = max(1, int(self.duration * temporal_mask_scale))

        # -- Sample spatial block mask scale
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = spatial_scale
        spatial_mask_scale = min_s + _rand * (max_s - min_s)
        spatial_num_keep = int(self.height * self.width * spatial_mask_scale)

        # -- Sample block aspect-ratio
        _rand = torch.rand(1, generator=generator).item()
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)

        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(spatial_num_keep * aspect_ratio)))
        w = int(round(math.sqrt(spatial_num_keep / aspect_ratio)))
        h = min(h, self.height)
        w = min(w, self.width)

        return (t, h, w)

    def _sample_block_mask(self, b_size):
        t, h, w = b_size
        top = torch.randint(0, self.height - h + 1, (1,))
        left = torch.randint(0, self.width - w + 1, (1,))
        start = torch.randint(0, self.duration - t + 1, (1,))

        mask = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
        mask[start:start+t, top:top+h, left:left+w] = 0

        # Context mask will only span the first X frames
        # (X=self.max_context_frames)
        if self.max_context_duration < self.duration:
            mask[self.max_context_duration:, :, :] = 0

        # --
        return mask

    def _generate_block_masks(self, batch_size):
        """
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample pred block size using seed
        # 2. sample several pred block locations for each image (w/o seed)
        # 3. return pred masks and complement (enc mask)
        """
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            temporal_scale=self.temporal_pred_mask_scale,
            spatial_scale=self.spatial_pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio,
        )

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_enc = min_keep_pred = self.duration * self.height * self.width
        for _ in range(batch_size):

            empty_context = True
            while empty_context:

                mask_e = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
                for _ in range(self.npred):
                    mask_e *= self._sample_block_mask(p_size)
                mask_e = mask_e.flatten()

                mask_p = torch.argwhere(mask_e == 0).squeeze()
                mask_e = torch.nonzero(mask_e).squeeze()

                empty_context = len(mask_e) == 0
                if not empty_context:
                    min_keep_pred = min(min_keep_pred, len(mask_p))
                    min_keep_enc = min(min_keep_enc, len(mask_e))
                    collated_masks_pred.append(mask_p)
                    collated_masks_enc.append(mask_e)

        if self.max_keep is not None:
            min_keep_enc = min(min_keep_enc, self.max_keep)

        collated_masks_pred = [cm[:min_keep_pred] for cm in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        collated_masks_enc = [cm[:min_keep_enc] for cm in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_masks_enc, collated_masks_pred
    
    def _generate_time_split_masks(self, video_batch):
        """
        video_batch: Tensor of shape (B, T, H, W) or (B, T, H, W, C)
        Returns:
            enc_mask: (B, N_enc) indices of encoder-visible patches
            pred_mask: (B, N_pred) indices of predictor-visible patches
        where masks are in patch space: (n_t, n_h, n_w)
        """
        B, T = video_batch.shape[0], video_batch.shape[1]
        
        # Make sure spatial_patch_size is a tuple
        if isinstance(self.spatial_patch_size, int):
            ph, pw = self.spatial_patch_size, self.spatial_patch_size
        else:
            ph, pw = self.spatial_patch_size
        pt = self.temporal_patch_size      # temporal patch size

        n_h, n_w = self.height, self.width   # patches per frame
        n_t = self.duration                  # temporal patches

        # What counts as "black"?
        black_value = video_batch.min().item()

        collated_masks_pred, collated_masks_enc = [], []
        print("Batch length:", len(batch))
        for b in batch:
            print(type(b))

        # Random temporal *patch* index for prediction
        t_patch = torch.randint(0, n_t - 1, (1,)).item()

        for b in range(B):
            # encoder mask in patch space
            enc_mask = torch.zeros((n_t, n_h, n_w), dtype=torch.int32)

            # go through all temporal patches up to t_patch
            for tt in range(t_patch + 1):  # inclusive
                t_start, t_end = tt * pt, (tt + 1) * pt
                for hh in range(n_h):
                    for ww in range(n_w):
                        h_start, h_end = hh * ph, (hh + 1) * ph
                        w_start, w_end = ww * pw, (ww + 1) * pw

                        patch = video_batch[b, t_start:t_end, h_start:h_end, w_start:w_end]
                        if not torch.all(patch == black_value):
                            enc_mask[tt, hh, ww] = 1

            # predictor mask: only the *next* temporal patch
            pred_mask = torch.zeros((n_t, n_h, n_w), dtype=torch.int32)
            pred_mask[t_patch + 1, :, :] = 1  # predict entire patch at t_patch+1
            pred_mask[t_patch + 1, :4, n_w-6:] = 0 # exclude the 6 rightmost columns corresponding to the compass
            
            # flatten to indices
            enc_mask = torch.nonzero(enc_mask.flatten()).squeeze()
            pred_mask = torch.nonzero(pred_mask.flatten()).squeeze()

            collated_masks_enc.append(enc_mask)
            collated_masks_pred.append(pred_mask)

        # Collate across batch
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_masks_enc, collated_masks_pred

    def __call__(self, batch):
        batch_size = len(batch)
        if self.mode == 'block':
            return self._generate_block_masks(batch_size)
        elif self.mode == 'time_split':
            return self._generate_time_split_masks(batch)
        else:
            raise ValueError(f'Invalid mode: {self.mode}')
