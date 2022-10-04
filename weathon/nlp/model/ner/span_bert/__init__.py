from weathon.nlp.dataset import SpanNERDataset as Dataset
from weathon.nlp.dataset import SpanNERDataset as SpanBertNERDataset


from weathon.nlp.processor.tokenizer import TokenTokenizer as Tokenizer
from weathon.nlp.processor.tokenizer import TokenTokenizer as SpanBertNERTokenizer

from weathon.nlp.nn import BertConfig as SpanBertConfig
from weathon.nlp.nn import BertConfig as ModuleConfig

from weathon.nlp.model.ner.span_bert.span_bert import  SpanIndependenceBert as SpanBert
from weathon.nlp.model.ner.span_bert.span_bert import  SpanIndependenceBert as Module

from weathon.utils.optimizer_utils import OptimizerUtils
get_default_model_optimizer = OptimizerUtils.get_default_bert_optimizer
get_default_span_bert_optimizer = OptimizerUtils.get_default_bert_optimizer

from weathon.nlp.model.ner.span_bert.span_bert_named_entity_recognition import SpanNERTask
from weathon.nlp.model.ner.span_bert.span_bert_named_entity_recognition import SpanNERTask as Task
from weathon.nlp.model.ner.span_bert.span_bert_named_entity_recognition import SpanNERTask as SpanBertNERTask

from weathon.nlp.predictor import SpanNERPredictor
from weathon.nlp.predictor import SpanNERPredictor as Predictor
from weathon.nlp.predictor import SpanNERPredictor as SpanBertNERPredictor

