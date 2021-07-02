import openpifpaf

from .freihand import Freihand
from .rhd import RHD
from .cifonly import CifOnly

def register():
    openpifpaf.DATAMODULES['freihand'] = Freihand
    openpifpaf.DATAMODULES['rhd'] = RHD
    openpifpaf.DECODERS.add(CifOnly)
#    openpifpaf.CHECKPOINT_URLS['shufflenetv2k16-wholebody'] = 'http://github.com/DuncanZauss/' \
#        'openpifpaf_assets/releases/download/v0.1.0/wb_shufflenet16_mixed_foot.pkl.epoch550'
   