import openpifpaf

from .freihand import Freihand
from .rhd import RHD
from .cifonly import CifOnly

def register():
    openpifpaf.DATAMODULES['freihand'] = Freihand
    openpifpaf.DATAMODULES['rhd'] = RHD
    openpifpaf.DECODERS.add(CifOnly)
    openpifpaf.CHECKPOINT_URLS['shufflenetv2k16-hand'] = 'https://github.com/DuncanZauss/' \
        'openpifpaf_assets/releases/download/v0.1.0/rhd_freihand_sk16.pkl.epoch600'
    openpifpaf.CHECKPOINT_URLS['shufflenetv2k16-wb-hand'] = 'https://github.com/DuncanZauss/' \
        'openpifpaf_assets/releases/download/v0.1.0/freihand_wholebody_sk16.pkl.epoch600'
   
