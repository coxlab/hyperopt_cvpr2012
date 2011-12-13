import cvpr_params
from lfw import LFWBandit
from cvpr_params import (null,
                         false,
                         true,
                         choice,
                         uniform,
                         gaussian,
                         lognormal,
                         qlognormal,
                         ref)


class LFWBanditReorder(LFWBandit):
    source_string = cvpr_params.string()
