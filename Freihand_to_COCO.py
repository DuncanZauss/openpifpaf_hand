import json
import numpy as np
import time
import os
from constants import FREIHAND_SKELETON, FREIHAND_KPS

def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg

def db_size(set_name):
    """ Hardcoded size of the datasets. """
    if set_name == 'training':
        return 32560  # number of unique samples (they exists in multiple 'versions')
    elif set_name == 'evaluation':
        return 3960
    else:
        assert 0, 'Invalid choice.'

def json_load(p):
    _assert_exist(p)
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d

def projectPoints(xyz, K):
    """ Project 3D coordinates into image space. """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]


def load_db_annotation(base_path, set_name=None):
    if set_name is None:
        # only training set annotations are released so this is a valid default choice
        set_name = 'training'

    print('Loading FreiHAND dataset index ...')
    t = time.time()

    # assumed paths to data containers
    k_path = os.path.join(base_path, '%s_K.json' % set_name)
    mano_path = os.path.join(base_path, '%s_mano.json' % set_name)
    xyz_path = os.path.join(base_path, '%s_xyz.json' % set_name)

    # load if exist
    K_list = json_load(k_path)
    mano_list = json_load(mano_path)
    xyz_list = json_load(xyz_path)

    # should have all the same length
    assert len(K_list) == len(mano_list), 'Size mismatch.'
    assert len(K_list) == len(xyz_list), 'Size mismatch.'

    print('Loading of %d samples done in %.2f seconds' % (len(K_list), time.time()-t))
    return list(zip(K_list, mano_list, xyz_list))

class sample_version:
    gs = 'gs'  # green screen
    hom = 'hom'  # homogenized
    sample = 'sample'  # auto colorization with sample points
    auto = 'auto'  # auto colorization without sample points: automatic color hallucination

    db_size = db_size('training')

    @classmethod
    def valid_options(cls):
        return [cls.gs, cls.hom, cls.sample, cls.auto]

    @classmethod
    def check_valid(cls, version):
        msg = 'Invalid choice: "%s" (must be in %s)' % (version, cls.valid_options())
        assert version in cls.valid_options(), msg

    @classmethod
    def map_id(cls, id, version):
        cls.check_valid(version)
        return id + cls.db_size*cls.valid_options().index(version)


def get_img_path(idx, base_path, set_name, version=None):
    if version is None:
        version = sample_version.gs

    if set_name == 'evaluation':
        assert version == sample_version.gs, 'This the only valid choice for samples from the evaluation split.'

    absolute_img_rgb_path = os.path.join(base_path, set_name, 'rgb',
                                '%08d.jpg' % sample_version.map_id(idx, version))
    _assert_exist(absolute_img_rgb_path)
    img_id = os.path.join('%08d.jpg' % sample_version.map_id(idx, version))
    return img_id

def main():
    # List of all the annotation types that should be used
   
    base_path = "../FreiHand"
    new_file = "../FreiHand/FreiHand_only_homogenized_Train_annotations_MSCOCO_style.json"

    # =============================================================================
    # orig_file = "../../data-mscoco/annotations_wholebody/coco_wholebody_val_v1.0.json"
    # new_file = "../../data-mscoco/annotations/"
    # "final_person_keypoints_val2017_wholebody_pifpaf_style.json"
    # =============================================================================
    new_data = {}
    new_data["info"] = dict(url="https://github.com/openpifpaf/openpifpaf",
        date_created=time.strftime("%a, %d %b %Y %H:%M:%S +0000",
                                   time.localtime()),
        description=("Conversion of FreiHand dataset into MS-COCO"))
    
    new_data["categories"] = [{"name":'hand',
                                 "id":1,
                                 "skeleton":list(FREIHAND_SKELETON),
                                 "supercategory":'hand',
                                 "keypoints":list(FREIHAND_KPS)}]
    new_data["images"] = []
    new_data["annotations"] = []

    # load annotations
    for folder_type in ['training']:
        db_data_anno = load_db_annotation(base_path, folder_type)
    
        # iterate over all samples
        for version in ["hom"]: #["gs", "hom", "sample", "auto"]:
            for idx in range(db_size(folder_type)):
                # annotation for this frame
                im_size = (224, 224)
                K, _, xyz = db_data_anno[idx]
                K, xyz = [np.array(x) for x in [K, xyz]]
                uv = projectPoints(xyz, K)
                box_tight = [np.min(uv[:, 0]), np.min(uv[:, 1]),
                     np.max(uv[:, 0]), np.max(uv[:, 1])]
                w, h = box_tight[2] - box_tight[0], box_tight[3] - box_tight[1]
                x_o = max(box_tight[0] - 0.1 * w, 0)
                y_o = max(box_tight[1] - 0.1 * h, 0)
                x_i = min(box_tight[0] + 1.1 * w, im_size[0])
                y_i = min(box_tight[1] + 1.1 * h, im_size[1])
                box = [int(x_o), int(y_o), int(x_i - x_o), int(y_i - y_o)]  # (x, y, w, h)
                
                im_name = get_img_path(idx, base_path, folder_type, version)
                
                uv = np.hstack((uv, 2 * np.ones((uv.shape[0], 1))))
                uv = np.vstack((uv, np.zeros((uv.shape[0], 3))))
                kps = np.array(uv, dtype = np.int).flatten().tolist()
                
                # Only one hand per image, so the number of hands = number of images
                new_data["annotations"].append({
                    'image_id': int(im_name[:-4]),
                    'category_id': 1,
                    'iscrowd': 0,
                    'id': int(im_name[:-4]),  # only one hand per image --> im_id=id, no need for unique treatment of different hands
                    'area': int(box[2] * box[3]),
                    'bbox': list(box),
                    'num_keypoints': 42,
                    'keypoints': kps,
                    'segmentation': []})
                
                new_data["images"].append({
                    'coco_url': "unknown",
                    'file_name': im_name,
                    'id': int(im_name[:-4]),  # unique id is the number from the image name
                    'license': 1,
                    'date_captured': "unknown",
                    'width': 224,
                    'height': 224}
                    )
    
    with open(new_file, 'w') as f:
        json.dump(new_data, f)

if __name__ == "__main__":
    main()
