import argparse
from collections import defaultdict
import heapq
import logging
import time
from typing import List

import numpy as np

from openpifpaf.decoder.decoder import Decoder
from openpifpaf.annotation import Annotation
from openpifpaf.decoder import utils
from openpifpaf import headmeta, visualizer

# pylint: disable=import-error
# from openpifpaf.functional import caf_center_s, grow_connection_blend

LOG = logging.getLogger(__name__)


class CifOnly(Decoder):
    """Generate CifOnly poses from fields.
    """
    force_complete = False
    keypoint_threshold = 0.15
    downsample_factor = 16

    def __init__(self,
                 cif_metas: List[headmeta.Cif],
                 *,
                 cif_visualizers=None):
        super().__init__()

        self.cif_metas = cif_metas
        self.keypoints = cif_metas[0].keypoints
        self.score_weights = cif_metas[0].score_weights
        self.out_skeleton = cif_metas[0].draw_skeleton

        self.cif_visualizers = cif_visualizers
        if self.cif_visualizers is None:
            self.cif_visualizers = [visualizer.Cif(meta) for meta in cif_metas]

        # prefer decoders with more keypoints and associations
        self.priority += sum(m.n_fields for m in cif_metas) / 1000.0

        self.timers = defaultdict(float)


    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        """Command line interface (CLI) to extend argument parser."""
        group = parser.add_argument_group('CifOnly decoder')
        assert not cls.force_complete
        group.add_argument('--cifonly-force-complete-pose',
                           default=False, action='store_true')
        group.add_argument('--cifonly-keypoint-threshold', type=float,
                           default=cls.keypoint_threshold,
                           help='filter keypoints by score')
        group.add_argument('--cifonly-without-highres',
                           default=False, action='store_true',
                           help="Create highres confidence maps and search for maxima in those")
        group.add_argument('--cifonly-downsample-factor', type=float,
                           default=cls.downsample_factor,
                           help='Ratio of the image to the output feature map size.')
        
        
    @classmethod
    def configure(cls, args: argparse.Namespace):
        """Take the parsed argument parser output and configure class variables."""
        cls.keypoint_threshold = args.cifonly_keypoint_threshold
        if args.cifonly_force_complete_pose:
            LOG.warn("Force complete pose is not recommended for the CifOnly decoder, " \
                     "it will force a prediction for any given image")
            cls.keypoint_threshold = 0.0
        cls.downsample_factor = args.cifonly_downsample_factor
        cls.without_highres = args.cifonly_without_highres
                 
    @classmethod
    def factory(cls, head_metas):
        return [
            CifOnly([meta])
            for meta in head_metas
            if (isinstance(meta, headmeta.Cif))
        ]

    def __call__(self, fields):
        start = time.perf_counter()
        for vis, meta in zip(self.cif_visualizers, self.cif_metas):
            vis.predicted(fields[meta.head_index])
        ann = Annotation(self.keypoints,
                          self.out_skeleton,
                          score_weights=self.score_weights)
        if not self.without_highres:
            # Slower decoder with proper high resolution maps
            cifhr = utils.CifHr().fill(fields, self.cif_metas)
            for f, hr_map in enumerate(cifhr.accumulated):
                y, x = np.unravel_index(hr_map.argmax(), hr_map.shape)
                v = np.max(hr_map)
                if v > self.keypoint_threshold:
                    ann.add(f, (x, y, v))
        else:
            # Slightly faster without high res maps
            cif_fields = fields[0]
            for f in range(cif_fields.shape[0]):
                kp_f = cif_fields[f , :, :]
                ind_i, ind_j = np.unravel_index(kp_f[1, :, :].argmax(), kp_f[1, :, :].shape)
                v = np.max(kp_f[1, :, :])
                x = kp_f[2, ind_i, ind_j] * self.downsample_factor  # TODO replace by stride/upsample 
                y = kp_f[3, ind_i, ind_j] * self.downsample_factor
                if v > self.keypoint_threshold:
                     ann.add(f, (x, y, v))
        annotations = [ann]
        LOG.debug('annotations %d, %.3fs', len(annotations), time.perf_counter() - start)
        LOG.info('%d annotations: %s', len(annotations),
                 [np.sum(ann.data[:, 2] > 0.1) for ann in annotations])
        return annotations