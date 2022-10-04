from weathon.nlp.dataset import GlobalPointerNERDataset as Dataset
from weathon.nlp.dataset import GlobalPointerNERDataset as GlobalPointerBertNERDataset

from weathon.nlp.processor.tokenizer import SpanTokenizer as Tokenizer
from weathon.nlp.processor.tokenizer import SpanTokenizer as GlobalPointerBertNERTokenizer

from weathon.nlp.nn import BertConfig as GlobalPointerBertConfig
from weathon.nlp.nn import BertConfig as ModuleConfig

from weathon.nlp.model.ner.global_pointer_bert.global_pointer_bert import GlobalPointerBert,EfficientGlobalPointer
from weathon.nlp.model.ner.global_pointer_bert.global_pointer_bert import GlobalPointerBert as Module

from weathon.utils.optimizer_utils import OptimizerUtils
get_default_model_optimizer = OptimizerUtils.get_default_bert_optimizer
get_default_global_pointer_bert_optimizer = OptimizerUtils.get_default_bert_optimizer

from weathon.nlp.model.ner.global_pointer_bert.global_pointer_bert_named_entity_recognition import GlobalPointerNERTask as Task
from weathon.nlp.model.ner.global_pointer_bert.global_pointer_bert_named_entity_recognition import GlobalPointerNERTask as GlobalPointerBertNERTask

from weathon.nlp.predictor import GlobalPointerNERPredictor as Predictor
from weathon.nlp.predictor import GlobalPointerNERPredictor as GlobalPointerBertNERPredictor

