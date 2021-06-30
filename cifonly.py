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
    """Generate CifCaf poses from fields.

    :param: nms: set to None to switch off non-maximum suppression.
    """
    occupancy_visualizer = visualizer.Occupancy()
    force_complete = False
    keypoint_threshold = 0.15

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
    
    @classmethod
    def configure(cls, args: argparse.Namespace):
        """Take the parsed argument parser output and configure class variables."""
        cls.keypoint_threshold = args.cifonly_keypoint_threshold
        if args.cifonly_force_complete_pose:
            LOG.warn("Force complete pose is not recommended for the CifOnly decoder, " \
                     "it will force a prediction for any given image")
            cls.keypoint_threshold = 0.0
                 
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
        cifhr = utils.CifHr().fill(fields, self.cif_metas)
        ann = Annotation(self.keypoints,
                          self.out_skeleton,
                          score_weights=self.score_weights)
        for f, hr_map in enumerate(cifhr.accumulated):
            y, x = np.unravel_index(hr_map.argmax(), hr_map.shape)
            v = np.max(hr_map)
            if v > self.keypoint_threshold:
                ann.add(f, (x, y, v)) 
                #ann.joint_scales[f] = 0.1 #s
# =============================================================================
#         hr_maps_stacked = np.array(cifhr.accumulated)
#         hr_shape = hr_maps_stacked.shape
#         coords = hr_maps_stacked.reshape((hr_shape[0], hr_shape[1] * hr_shape[2])).argmax(axis=1)
#         x_coords = coords % hr_shape[1]
#         y_coords = coords / hr_shape[2]
#         v_l = hr_maps_stacked.max(axis=(1,2))
#         for f, (x, y, v) in enumerate(zip(x_coords, y_coords, v_l)):
#             #print(f, len(x), len(y), len(v))
#             if v > self.keypoint_threshold:
#                 ann.add(f, (x, y, v)) 
#                 #ann.joint_scales[f] = 0.1 #s
# =============================================================================
        annotations = [ann]
        LOG.debug('annotations %d, %.3fs', len(annotations), time.perf_counter() - start)
        LOG.info('%d annotations: %s', len(annotations),
                 [np.sum(ann.data[:, 2] > 0.1) for ann in annotations])
        return annotations