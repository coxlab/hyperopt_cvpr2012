from starflow.protocols import Apply, Applies, protocolize

import deploy
import main_protocols as protocols

from cvpr_params import model, model_activ_hetero           


"""
In this module we will:
    -- call protocols.image_protocol to set up runs to create image sets in the DB
    -- call protocols.model_protocol to set up runs to create models in the DB
    -- call protcols.optimization_protocol, which will set up a hyperopt-mongo search process, 
         the bandits will be prepared using genson source strings created from cvpr_parmas (imported above)
"""

