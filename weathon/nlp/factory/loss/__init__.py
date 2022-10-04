from torch.nn.modules.loss import BCEWithLogitsLoss, CrossEntropyLoss, SmoothL1Loss

from weathon.nlp.factory.loss.casrel_loss import CasRelLoss
from weathon.nlp.factory.loss.focal_loss import FocalLoss
from weathon.nlp.factory.loss.label_smoothing_ce_loss import LabelSmoothingCrossEntropy
from weathon.nlp.factory.loss.global_pointer_ce_loss import GlobalPointerCrossEntropy
from weathon.nlp.factory.loss.r_drop_cross_entropy_loss import RDropCrossEntropyLoss

all_losses_dict = dict(
    binarycrossentropy=BCEWithLogitsLoss,
    bce=BCEWithLogitsLoss,
    crossentropy=CrossEntropyLoss,
    ce=CrossEntropyLoss,
    smoothl1=SmoothL1Loss,
    casrel=CasRelLoss,
    focal=FocalLoss,
    labelsmoothingcrossentropy=LabelSmoothingCrossEntropy,
    lsce=LabelSmoothingCrossEntropy,
    gpce=GlobalPointerCrossEntropy,
    r_dropce=RDropCrossEntropyLoss
)
