from weathon.nlp.dataset import PairMergeSentenceClassificationDataset as Dataset
from weathon.nlp.dataset import PairMergeSentenceClassificationDataset as BertTMDataset

from weathon.nlp.processor.tokenizer import PairTokenizer as Tokenizer
from weathon.nlp.processor.tokenizer import PairTokenizer as BertTMTokenizer

from weathon.nlp.nn import BertConfig
from weathon.nlp.nn import BertConfig as ModuleConfig

from weathon.nlp.nn import Bert
from weathon.nlp.nn import Bert as Module

from weathon.utils.optimizer_utils import OptimizerUtils

get_default_model_optimizer = OptimizerUtils.get_default_bert_optimizer
get_default_bert_optimizer = OptimizerUtils.get_default_bert_optimizer

from weathon.nlp.task import TMTask as Task
from weathon.nlp.task import TMTask as BertTMTask

from weathon.nlp.predictor import TMPredictor as Predictor
from weathon.nlp.predictor import TMPredictor as BertTMPredictor
