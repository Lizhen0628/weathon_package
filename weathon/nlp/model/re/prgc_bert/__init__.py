from weathon.nlp.model.re.prgc_bert.prgc_relation_extraction_dataset import PRGCREDataset
from weathon.nlp.model.re.prgc_bert.prgc_relation_extraction_dataset import PRGCREDataset as Dataset

from weathon.nlp.processor.tokenizer import SpanTokenizer as Tokenizer
from weathon.nlp.processor.tokenizer import SpanTokenizer as PRGCRETokenizer

from weathon.nlp.nn import BertConfig as PRGCBertConfig
from weathon.nlp.nn import BertConfig as ModuleConfig

from weathon.nlp.model.re.prgc_bert.prgc_bert import PRGCBert
from weathon.nlp.model.re.prgc_bert.prgc_bert import PRGCBert as Module

from weathon.utils.optimizer_utils import OptimizerUtils

get_default_model_optimizer = OptimizerUtils.get_default_bert_optimizer
get_default_prgc_bert_optimizer = OptimizerUtils.get_default_bert_optimizer

from weathon.nlp.model.re.prgc_bert.prgc_relation_extraction_task import PRGCRETask as Task
from weathon.nlp.model.re.prgc_bert.prgc_relation_extraction_task import PRGCRETask as PRGCRETask

from weathon.nlp.model.re.prgc_bert.prgc_relation_extraction_predictor import PRGCREPredictor as Predictor
from weathon.nlp.model.re.prgc_bert.prgc_relation_extraction_predictor import PRGCREPredictor as PRGCREPredictor
