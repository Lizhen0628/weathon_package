from weathon.nlp.dataset import SentenceClassificationDataset as Dataset
from weathon.nlp.dataset import SentenceClassificationDataset as ErnieTCDataset

from weathon.nlp.processor.tokenizer import SentenceTokenizer as Tokenizer
from weathon.nlp.processor.tokenizer import SentenceTokenizer as ErnieTCTokenizer

from weathon.nlp.nn import ErnieConfig
from weathon.nlp.nn import ErnieConfig as ModuleConfig

from weathon.nlp.nn import Ernie
from weathon.nlp.nn import Ernie as Module

from weathon.utils.optimizer_utils import OptimizerUtils

get_default_model_optimizer = OptimizerUtils.get_default_bert_optimizer
get_default_ernie_optimizer = OptimizerUtils.get_default_bert_optimizer

from weathon.nlp.task import SequenceClassificationTask as Task
from weathon.nlp.task import SequenceClassificationTask as ErnieTCTask

from weathon.nlp.predictor import TCPredictor as Predictor
from weathon.nlp.predictor import TCPredictor as ErnieTCPredictor
