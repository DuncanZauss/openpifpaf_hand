import argparse
import torch
import openpifpaf

from openpifpaf.plugins.coco import CocoDataset as Coco
#from .wholebody_metric import WholebodyMetric
from .constants import (
    FREIHAND_SKELETON, FREIHAND_POSE, FREIHAND_SCORE_WEIGHTS, FREIHAND_KPS, FREIHAND_SIGMAS,
    FREIHAND_CATEGORIES, HFLIP
    )

try:
    import pycocotools.coco
    # monkey patch for Python 3 compat
    pycocotools.coco.unicode = str
except ImportError:
    pass

class RHD(openpifpaf.datasets.DataModule):
    # cli configurable
    train_annotations = 'Rendered_Hand_Dataset/RHD_training_annotations_MSCOCO_style.json'
    val_annotations = 'Rendered_Hand_Dataset/RHD_evaluation_annotations_MSCOCO_style.json'
    train_image_dir = "Rendered_Hand_Dataset/training/color"
    val_image_dir = "Rendered_Hand_Dataset/evaluation/color" 

    square_edge = 385
    extended_scale = False
    orientation_invariant = 0.0
    blur = 0.0
    augmentation = True
    rescale_images = 1.0
    upsample_stride = 1
    min_kp_anns = 1
    bmin = 1.0
    train_only_cif = False

    eval_annotation_filter = True
    eval_long_edge = 641
    eval_orientation_invariant = 0.0
    eval_extended_scale = False

    def __init__(self):
        super().__init__()
        cif = openpifpaf.headmeta.Cif('cif', 'rhd',
                                      keypoints=FREIHAND_KPS,
                                      sigmas=FREIHAND_SIGMAS,
                                      pose=FREIHAND_POSE,
                                      draw_skeleton=FREIHAND_SKELETON,
                                      score_weights=FREIHAND_SCORE_WEIGHTS)
        caf = openpifpaf.headmeta.Caf('caf', 'rhd',
                                      keypoints=FREIHAND_KPS,
                                      sigmas=FREIHAND_SIGMAS,
                                      pose=FREIHAND_POSE,
                                      skeleton=FREIHAND_SKELETON,)

        cif.upsample_stride = self.upsample_stride
        caf.upsample_stride = self.upsample_stride
        self.head_metas = [cif] if self.train_only_cif else [cif, caf]

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module rhd')

        group.add_argument('--rhd-train-annotations', default=cls.train_annotations,
                           help='train annotations')
        group.add_argument('--rhd-val-annotations', default=cls.val_annotations,
                           help='val annotations')
        group.add_argument('--rhd-train-image-dir', default=cls.train_image_dir,
                           help='train image dir')
        group.add_argument('--rhd-val-image-dir', default=cls.val_image_dir,
                           help='val image dir')

        group.add_argument('--rhd-square-edge',
                           default=cls.square_edge, type=int,
                           help='square edge of input images')
        assert not cls.extended_scale
        group.add_argument('--rhd-extended-scale',
                           default=False, action='store_true',
                           help='augment with an extended scale range')
        group.add_argument('--rhd-orientation-invariant',
                           default=cls.orientation_invariant, type=float,
                           help='augment with random orientations')
        group.add_argument('--rhd-blur',
                           default=cls.blur, type=float,
                           help='augment with blur')
        assert cls.augmentation
        group.add_argument('--rhd-no-augmentation',
                           dest='rhd_augmentation',
                           default=True, action='store_false',
                           help='do not apply data augmentation')
        group.add_argument('--rhd-rescale-images',
                           default=cls.rescale_images, type=float,
                           help='overall rescale factor for images')
        group.add_argument('--rhd-upsample',
                           default=cls.upsample_stride, type=int,
                           help='head upsample stride')
        group.add_argument('--rhd-min-kp-anns',
                           default=cls.min_kp_anns, type=int,
                           help='filter images with fewer keypoint annotations')
        group.add_argument('--rhd-bmin',
                           default=cls.bmin, type=float,
                           help='bmin')
        group.add_argument('--rhd-train-only-cif-heads',
                           dest='rhd_train_only_cif_heads',
                           default=False, action='store_true',
                           help='Train only a CIF head, which can be used for the CifOnly decoder.')

        # evaluation
        eval_set_group = group.add_mutually_exclusive_group()
        eval_set_group.add_argument('--rhd-eval-test2017', default=False,
                                    action='store_true')
        eval_set_group.add_argument('--rhd-eval-testdev2017', default=False,
                                    action='store_true')

        assert cls.eval_annotation_filter
        group.add_argument('--rhd-no-eval-annotation-filter',
                           dest='rhd_eval_annotation_filter',
                           default=True, action='store_false')
        group.add_argument('--rhd-eval-long-edge', default=cls.eval_long_edge, type=int,
                           help='set to zero to deactivate rescaling')
        assert not cls.eval_extended_scale
        group.add_argument('--rhd-eval-extended-scale', default=False, action='store_true')
        group.add_argument('--rhd-eval-orientation-invariant',
                           default=cls.eval_orientation_invariant, type=float)

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        cls.train_annotations = args.rhd_train_annotations
        cls.val_annotations = args.rhd_val_annotations
        cls.eval_annotations = cls.val_annotations
        cls.train_image_dir = args.rhd_train_image_dir
        cls.val_image_dir = args.rhd_val_image_dir
        cls.eval_image_dir = cls.val_image_dir

        cls.square_edge = args.rhd_square_edge
        cls.extended_scale = args.rhd_extended_scale
        cls.orientation_invariant = args.rhd_orientation_invariant
        cls.blur = args.rhd_blur
        cls.augmentation = args.rhd_augmentation
        cls.rescale_images = args.rhd_rescale_images
        cls.upsample_stride = args.rhd_upsample
        cls.min_kp_anns = args.rhd_min_kp_anns
        cls.bmin = args.rhd_bmin
        cls.train_only_cif = args.rhd_train_only_cif_heads

        # evaluation
        cls.eval_annotation_filter = args.rhd_eval_annotation_filter
        cls.eval_long_edge = args.rhd_eval_long_edge
        cls.eval_orientation_invariant = args.rhd_eval_orientation_invariant
        cls.eval_extended_scale = args.rhd_eval_extended_scale


    def _preprocess(self):
        if self.train_only_cif:
            encoders = [openpifpaf.encoder.Cif(self.head_metas[0], bmin=self.bmin)]
        else:
            encoders = (openpifpaf.encoder.Cif(self.head_metas[0], bmin=self.bmin),
                        openpifpaf.encoder.Caf(self.head_metas[1], bmin=self.bmin))

        if not self.augmentation:
            return openpifpaf.transforms.Compose([
                openpifpaf.transforms.NormalizeAnnotations(),
                openpifpaf.transforms.RescaleAbsolute(self.square_edge),
                openpifpaf.transforms.CenterPad(self.square_edge),
                openpifpaf.transforms.EVAL_TRANSFORM,
                openpifpaf.transforms.Encoders(encoders),
            ])

        if self.extended_scale:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.25 * self.rescale_images,
                             2.0 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))
        else:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.4 * self.rescale_images,
                             2.0 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))

        return openpifpaf.transforms.Compose([
            openpifpaf.transforms.NormalizeAnnotations(),
            openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.HFlip(FREIHAND_KPS, HFLIP), 0.5),
            rescale_t,
            openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.Blur(), self.blur),
            openpifpaf.transforms.Crop(self.square_edge, use_area_of_interest=True),
            openpifpaf.transforms.CenterPad(self.square_edge),
            openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.RotateBy90(), self.orientation_invariant),
            openpifpaf.transforms.TRAIN_TRANSFORM,
            openpifpaf.transforms.Encoders(encoders),
        ])

    def train_loader(self):
        train_data = Coco(
            image_dir=self.train_image_dir,
            ann_file=self.train_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[1],
        )
        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=not self.debug and self.augmentation,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    def val_loader(self):
        val_data = Coco(
            image_dir=self.val_image_dir,
            ann_file=self.val_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[1],
        )
        return torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    @classmethod
    def common_eval_preprocess(cls):
        rescale_t = None
        if cls.eval_extended_scale:
            assert cls.eval_long_edge
            rescale_t = [
                openpifpaf.transforms.DeterministicEqualChoice([
                    openpifpaf.transforms.RescaleAbsolute(cls.eval_long_edge),
                    openpifpaf.transforms.RescaleAbsolute((cls.eval_long_edge - 1) // 2 + 1),
                ], salt=1)
            ]
        elif cls.eval_long_edge:
            rescale_t = openpifpaf.transforms.RescaleAbsolute(cls.eval_long_edge)

        if cls.batch_size == 1:
            padding_t = openpifpaf.transforms.CenterPadTight(16)
        else:
            assert cls.eval_long_edge
            padding_t = openpifpaf.transforms.CenterPad(cls.eval_long_edge)

        orientation_t = None
        if cls.eval_orientation_invariant:
            orientation_t = openpifpaf.transforms.DeterministicEqualChoice([
                None,
                openpifpaf.transforms.RotateBy90(fixed_angle=90),
                openpifpaf.transforms.RotateBy90(fixed_angle=180),
                openpifpaf.transforms.RotateBy90(fixed_angle=270),
            ], salt=3)

        return [
            openpifpaf.transforms.NormalizeAnnotations(),
            rescale_t,
            padding_t,
            orientation_t,
        ]

    def _eval_preprocess(self):
        return openpifpaf.transforms.Compose([
            *self.common_eval_preprocess(),
            openpifpaf.transforms.ToAnnotations([
                openpifpaf.transforms.ToKpAnnotations(
                    FREIHAND_CATEGORIES,
                    keypoints_by_category={1: self.head_metas[0].keypoints},
                    skeleton_by_category={1: self.head_metas[0].draw_skeleton},
                ),
                openpifpaf.transforms.ToCrowdAnnotations(FREIHAND_CATEGORIES),
            ]),
            openpifpaf.transforms.EVAL_TRANSFORM,
        ])

    def eval_loader(self):
        eval_data = Coco(
            image_dir=self.eval_image_dir,
            ann_file=self.eval_annotations,
            preprocess=self._eval_preprocess(),
            annotation_filter=self.eval_annotation_filter,
            min_kp_anns=self.min_kp_anns if self.eval_annotation_filter else 0,
            category_ids=[1] if self.eval_annotation_filter else [],
        )
        return torch.utils.data.DataLoader(
            eval_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=False,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta)

    def metrics(self):
        return []
