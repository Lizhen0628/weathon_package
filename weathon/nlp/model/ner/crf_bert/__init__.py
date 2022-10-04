from weathon.nlp.dataset import BIONERDataset as Dataset
from weathon.nlp.dataset import BIONERDataset as CrfBertNERDataset

from weathon.nlp.processor.tokenizer import TokenTokenizer as Tokenizer
from weathon.nlp.processor.tokenizer import TokenTokenizer as CrfBertNERTokenizer

from weathon.nlp.nn import BertConfig as CrfBertConfig
from weathon.nlp.nn import BertConfig as ModuleConfig

from weathon.nlp.nn import CrfBert
from weathon.nlp.nn import CrfBert as Module

from weathon.utils.optimizer_utils import OptimizerUtils

get_default_model_optimizer = OptimizerUtils.get_default_bert_optimizer
get_default_crf_bert_optimizer = OptimizerUtils.get_default_bert_optimizer

from weathon.nlp.task import CRFNERTask as Task
from weathon.nlp.task import CRFNERTask as CrfBertNERTask

from weathon.nlp.predictor import CRFNERPredictor as Predict
from weathon.nlp.predictor import CRFNERPredictor as CrfBertNERPredictor
