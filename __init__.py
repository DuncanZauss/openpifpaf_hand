import openpifpaf

from .freihand import Freihand
from .freihand_fast import FreihandFast
from .cifonly import CifOnly

def register():
    openpifpaf.DATAMODULES['freihand'] = Freihand
    openpifpaf.DATAMODULES['freihandfast'] = FreihandFast
    openpifpaf.DECODERS.add(CifOnly)
#    openpifpaf.CHECKPOINT_URLS['shufflenetv2k16-wholebody'] = 'http://github.com/DuncanZauss/' \
#        'openpifpaf_assets/releases/download/v0.1.0/wb_shufflenet16_mixed_foot.pkl.epoch550'
   