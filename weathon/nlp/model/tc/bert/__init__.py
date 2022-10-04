from weathon.nlp.dataset import SentenceClassificationDataset as Dataset
from weathon.nlp.dataset import SentenceClassificationDataset as BertTCDataset

from weathon.nlp.processor.tokenizer import SentenceTokenizer as Tokenizer
from weathon.nlp.processor.tokenizer import SentenceTokenizer as BertTCTokenizer

from weathon.nlp.nn import BertConfig
from weathon.nlp.nn import BertConfig as ModuleConfig

from weathon.nlp.nn import Bert
from weathon.nlp.nn import Bert as Module

from weathon.utils.optimizer_utils import OptimizerUtils

get_default_model_optimizer = OptimizerUtils.get_default_bert_optimizer
get_default_bert_optimizer = OptimizerUtils.get_default_bert_optimizer

from weathon.nlp.task import TCTask as Task
from weathon.nlp.task import TCTask as BertTCTask

from weathon.nlp.predictor import TCPredictor as Predictor
from weathon.nlp.predictor import TCPredictor as BertTCPredictor
