from weathon.nlp.factory.optimizer.adafactor import Adafactor
from weathon.nlp.factory.optimizer.madgrad import MADGRAD
from weathon.nlp.factory.optimizer.prior_wd import PriorWD
from torch.optim import Adadelta, Adagrad, Adam,AdamW, SparseAdam, Adamax, ASGD, LBFGS, RMSprop, Rprop, SGD

all_optimizers_dict = dict(
    adadelta=Adadelta,
    adagrad=Adagrad,
    adam=Adam,
    sparseadam=SparseAdam,
    adamax=Adamax,
    asgd=ASGD,
    lbfgs=LBFGS,
    rmsprop=RMSprop,
    rprop=Rprop,
    sgd=SGD,
    adamw=AdamW,
    adafactor=Adafactor,
    madgrad=MADGRAD,
    prior_wd=PriorWD
)
