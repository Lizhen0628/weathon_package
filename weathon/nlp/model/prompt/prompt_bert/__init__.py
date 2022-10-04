from weathon.nlp.dataset import PromptDataset as Dataset
from weathon.nlp.dataset import PromptDataset as PromptBertDataset

from weathon.nlp.processor.tokenizer import PromptMLMTransformerTokenizer
from weathon.nlp.processor.tokenizer import PromptMLMTransformerTokenizer as Tokenizer
from weathon.nlp.processor.tokenizer import PromptMLMTransformerTokenizer as PromptBertTokenizer

from weathon.nlp.nn import BertConfig as PromptBertConfig
from weathon.nlp.nn import BertConfig as ModuleConfig

from weathon.nlp.nn import BertForPromptMaskedLM as PromptBert
from weathon.nlp.nn import BertForPromptMaskedLM as Module

from weathon.utils.optimizer_utils import OptimizerUtils

get_default_model_optimizer = OptimizerUtils.get_default_bert_optimizer
get_default_prompt_bert_optimizer = OptimizerUtils.get_default_bert_optimizer

from weathon.nlp.task import PromptMLMTask as Task
from weathon.nlp.task import PromptMLMTask as PromptBertMLMTask

from weathon.nlp.predictor import PromptMLMPredictor as Predictor
from weathon.nlp.predictor import PromptMLMPredictor as PromptBertMLMPredictor
