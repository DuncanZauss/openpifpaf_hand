import numpy as np
import copy
# =============================================================================
# from openpifpaf.plugins.wholebody.constants import (
#     righthand_skeleton, righthand_pose, righthand,
#     # lefthand_skeleton, lefthand_pose, lefthand,
#     )
# 
# =============================================================================

righthand_skeleton = np.array([
    (113, 114), (113, 118), (113, 122), (113, 126),   # connect to finger starts
    (113, 130)] + [(x, x + 1) for s in [114, 118, 122, 126, 130] for x in range(s, s + 3)]
    + [(115, 118), (118, 122), (122, 126), (126, 130)]
    ) - 112
righthand_skeleton = righthand_skeleton.tolist()

lefthand_skeleton = np.array([
    (92, 93), (92, 97), (92, 101), (92, 105),  # connect to finger starts
    (92, 109)] + [(x, x + 1) for s in [93, 97, 101, 105, 109] for x in range(s, s + 3)]
    + [(94, 97), (97, 101), (101, 105), (105, 109)]
    ) - 70
lefthand_skeleton = lefthand_skeleton.tolist()

lefthand_pose = np.array([
    [-1.75, 3.9, 2.0],  # 92
    [-1.65, 3.8, 2.0],  # 93
    [-1.55, 3.7, 2.0],  # 94
    [-1.45, 3.6, 2.0],  # 95
    [-1.35, 3.5, 2.0],  # 96
    [-1.6, 3.5, 2.0],   # 97
    [-1.566, 3.4, 2.0],  # 98
    [-1.533, 3.3, 2.0],  # 99
    [-1.5, 3.2, 2.0],   # 100
    [-1.75, 3.5, 2.0],  # 101
    [-1.75, 3.4, 2.0],  # 102
    [-1.75, 3.3, 2.0],  # 103
    [-1.75, 3.2, 2.0],  # 104
    [-1.9, 3.5, 2.0],   # 105
    [-1.933, 3.4, 2.0],  # 106
    [-1.966, 3.3, 2.0],  # 107
    [-2.0, 3.2, 2.0],   # 108
    [-2.1, 3.5, 2.0],   # 109
    [-2.133, 3.433, 2.0],   # 110
    [-2.166, 3.366, 2.0],   # 111
    [-2.2, 3.3, 2.0], ])      # 112

lefthand_pose[:, 0] = (lefthand_pose[:, 0] + 1.75) * 1.0 - 2.25
lefthand_pose[:, 1] = (lefthand_pose[:, 1] - 3.9) * 1.5 + 4.4

righthand_pose = copy.deepcopy(lefthand_pose)
righthand_pose[:, 0] = -lefthand_pose[:, 0]

righthand = [0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025,
             0.024, 0.035, 0.018, 0.024, 0.022, 0.026, 0.017,
             0.021, 0.021, 0.032, 0.02, 0.019, 0.022, 0.031]

lefthand = [0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025,
            0.024, 0.035, 0.018, 0.024, 0.022, 0.026, 0.017,
            0.021, 0.021, 0.032, 0.02, 0.019, 0.022, 0.031]

righth_kps = ['rh_' + str(x) for x in range(0, 21)]
lefth_kps = ['lh_' + str(x) for x in range(21, 42)]

HFLIP = {
    'lh_21': 'rh_0',
    'rh_0': 'lh_21',
    'lh_22': 'rh_1',
    'rh_1': 'lh_22',
    'lh_23': 'rh_2',
    'rh_2': 'lh_23',
    'lh_24': 'rh_3',
    'rh_3': 'lh_24',
    'lh_25': 'rh_4',
    'rh_4': 'lh_25',
    'lh_26': 'rh_5',
    'rh_5': 'lh_26',
    'lh_27': 'rh_6',
    'rh_6': 'lh_27',
    'lh_28': 'rh_7',
    'rh_7': 'lh_28',
    'lh_29': 'rh_8',
    'rh_8': 'lh_29',
    'lh_30': 'rh_9',
    'rh_9': 'lh_30',
    'lh_31': 'rh_10',
    'rh_10': 'lh_31',
    'lh_32': 'rh_11',
    'rh_11': 'lh_32',
    'lh_33': 'rh_12',
    'rh_12': 'lh_33',
    'lh_34': 'rh_13',
    'rh_13': 'lh_34',
    'lh_35': 'rh_14',
    'rh_14': 'lh_35',
    'lh_36': 'rh_15',
    'rh_15': 'lh_36',
    'lh_37': 'rh_16',
    'rh_16': 'lh_37',
    'lh_38': 'rh_17',
    'rh_17': 'lh_38',
    'lh_39': 'rh_18',
    'rh_18': 'lh_39',
    'lh_40': 'rh_19',
    'rh_19': 'lh_40',
    'lh_41': 'rh_20',
    'rh_20': 'lh_41'
}

FREIHAND_CATEGORIES = "hand"

RIGHTHAND_SCORE_WEIGHTS = [100.0] * 3 + [1.0] * (len(righth_kps) - 3)

FREIHAND_SCORE_WEIGHTS = [100.0] * 3 + [1.0] * (len(righth_kps) - 3) + [0.0] * len(righth_kps)
FREIHAND_SKELETON = righthand_skeleton + lefthand_skeleton
FREIHAND_POSE = np.vstack((righthand_pose, lefthand_pose))
FREIHAND_KPS = righth_kps + lefth_kps
FREIHAND_SIGMAS = righthand + lefthand


def draw_ann(ann, *, keypoint_painter, filename=None, margin=0.5, aspect=None, **kwargs):
    from openpifpaf import show  # pylint: disable=import-outside-toplevel

    bbox = ann.bbox()
    xlim = bbox[0] - margin, bbox[0] + bbox[2] + margin
    ylim = bbox[1] - margin, bbox[1] + bbox[3] + margin
    if aspect == 'equal':
        fig_w = 5.0
    else:
        fig_w = 5.0 / (ylim[1] - ylim[0]) * (xlim[1] - xlim[0])

    with show.canvas(filename, figsize=(fig_w, 5), nomargin=True, **kwargs) as ax:
        ax.set_axis_off()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        if aspect is not None:
            ax.set_aspect(aspect)

        keypoint_painter.annotation(ax, ann)


def draw_skeletons(pose, prefix=""):
    from openpifpaf.annotation import Annotation  # pylint: disable=import-outside-toplevel
    from openpifpaf import show  # pylint: disable=import-outside-toplevel

    scale = np.sqrt(
        (np.max(pose[:, 0]) - np.min(pose[:, 0]))
        * (np.max(pose[:, 1]) - np.min(pose[:, 1]))
    )

    show.KeypointPainter.show_joint_scales = True
    keypoint_painter = show.KeypointPainter(line_width=2)

    ann = Annotation(keypoints=FREIHAND_KPS,
                     skeleton=FREIHAND_SKELETON,
                     score_weights=FREIHAND_SCORE_WEIGHTS)
    ann.set(pose, np.array(FREIHAND_SIGMAS) * scale)
    draw_ann(ann, filename='./docs/' + prefix + 'skeleton_FREIHAND.png',
             keypoint_painter=keypoint_painter)


def print_associations():
    for j1, j2 in FREIHAND_SKELETON:
        print(FREIHAND_KPS[j1 - 1], '-', FREIHAND_KPS[j2 - 1])


def rotate(pose, angle=45, axis=2):
    sin = np.sin(np.radians(angle))
    cos = np.cos(np.radians(angle))
    pose_copy = np.copy(pose)
    pose_copy[:, 2] = pose_copy[:, 2] - 2  # COOS at human center
    if axis == 0:
        rot_mat = np.array([[1, 0, 0],
                            [0, cos, -sin],
                            [0, sin, cos]])
    elif axis == 1:
        rot_mat = np.array([[cos, 0, sin],
                            [0, 1, 0],
                            [-sin, 0, cos]])
    elif axis == 2:
        rot_mat = np.array([[cos, -sin, 0],
                            [sin, cos, 0],
                            [0, 0, 1]])
    else:
        raise Exception("Axis must be 0,1 or 2 (corresponding to x,y,z).")
    rotated_pose = np.transpose(np.matmul(rot_mat, np.transpose(pose_copy)))
    rotated_pose[:, 2] = rotated_pose[:, 2] + 7  # assure sufficient depth for plotting
    return rotated_pose


if __name__ == '__main__':
    print_associations()
    draw_skeletons(FREIHAND_POSE)
