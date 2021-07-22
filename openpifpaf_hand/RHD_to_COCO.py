from __future__ import print_function, unicode_literals

import pickle
import os
import numpy as np
import json
import time
from constants import FREIHAND_SKELETON, FREIHAND_KPS

def main():
    base_path = "../Rendered_Hand_Dataset"            

    rhd_to_freihand_kp_matching = [21, 25, 24, 23, 22, 29, 28, 27, 26, 33, 32, 31, 30, # left hand
                                   37, 36, 35, 34, 41, 40, 39, 38, 
                                   0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, #Right hand
                                   16, 15, 14, 13, 20, 19, 18, 17]
    assert len(rhd_to_freihand_kp_matching) == 42
    assert len(set(rhd_to_freihand_kp_matching)) == 42

    for d_set in ['training', 'evaluation']:
        new_data = {}
        new_data["info"] = dict(url="https://github.com/openpifpaf/openpifpaf",
            date_created=time.strftime("%a, %d %b %Y %H:%M:%S +0000",
                                       time.localtime()),
            description=("Conversion of RHD dataset into MS-COCO"))
        
        new_data["categories"] = [{"name":'hand',
                                     "id":1,
                                     "skeleton":list(FREIHAND_SKELETON),
                                     "supercategory":'hand',
                                     "keypoints":list(FREIHAND_KPS)}]
        new_data["images"] = []
        new_data["annotations"] = []
        
        new_file = "../Rendered_Hand_Dataset/RHD_" + d_set + "_annotations_MSCOCO_style.json"

        # load annotations of this set
        with open(os.path.join(base_path, d_set, 'anno_%s.pickle' % d_set), 'rb') as fi:
            anno_all = pickle.load(fi)
        
        # iterate samples of the set
        for sample_id, anno in anno_all.items():
            # load data
            im_name = '%.5d.png' % sample_id
            
            # get info from annotation dictionary
            uv = anno['uv_vis']
            uv = uv[rhd_to_freihand_kp_matching]
            for jj in range(uv.shape[0]):
                if uv[jj, 2]==0:
                    uv[jj, 0]=0
                    uv[jj, 1]=0
            im_size = (320, 320)
            box_tight = [np.min(uv[:, 0]), np.min(uv[:, 1]),
                  np.max(uv[:, 0]), np.max(uv[:, 1])]
            w, h = box_tight[2] - box_tight[0], box_tight[3] - box_tight[1]
            x_o = max(box_tight[0] - 0.1 * w, 0)
            y_o = max(box_tight[1] - 0.1 * h, 0)
            x_i = min(box_tight[0] + 1.1 * w, im_size[0])
            y_i = min(box_tight[1] + 1.1 * h, im_size[1])
            box = [int(x_o), int(y_o), int(x_i - x_o), int(y_i - y_o)]  # (x, y, w, h)
            uv_flattened = uv.flatten().tolist()
            # Only one hand per image, so the number of hands = number of images
            new_data["annotations"].append({
                'image_id': int(im_name[:-4]),
                'category_id': 1,
                'iscrowd': 0,
                'id': int(im_name[:-4]),  # only one hand per image --> im_id=id, no need for unique treatment of different hands
                'area': int(box[2] * box[3]),
                'bbox': list(box),
                'num_keypoints': 42,
                'keypoints': uv_flattened,
                'segmentation': []})
            
            new_data["images"].append({
                'coco_url': "unknown",
                'file_name': im_name,
                'id': int(im_name[:-4]),  # unique id is the number from the image name
                'license': 1,
                'date_captured': "unknown",
                'width': 320,
                'height': 320}
                )
    
        with open(new_file, 'w') as f:
            json.dump(new_data, f)

if __name__ == "__main__":
    main()
