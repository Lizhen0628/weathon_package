from weathon.nlp.model.re.casrel_bert.casrel_relation_extraction_dataset import CasRelREDataset
from weathon.nlp.model.re.casrel_bert.casrel_relation_extraction_dataset import CasRelREDataset as Dataset

from weathon.nlp.processor.tokenizer import SpanTokenizer as Tokenizer
from weathon.nlp.processor.tokenizer import SpanTokenizer as CasRelRETokenizer

from weathon.nlp.nn import BertConfig as CasRelBertConfig
from weathon.nlp.nn import BertConfig as ModuleConfig

from weathon.nlp.model.re.casrel_bert.casrel_bert import CasRelBert
from weathon.nlp.model.re.casrel_bert.casrel_bert import CasRelBert as Module

from weathon.utils.optimizer_utils import OptimizerUtils

get_default_model_optimizer = OptimizerUtils.get_default_bert_optimizer
get_default_casrel_bert_optimizer = OptimizerUtils.get_default_bert_optimizer

from weathon.nlp.model.re.casrel_bert.casrel_relation_extraction_task import CasRelRETask as Task
from weathon.nlp.model.re.casrel_bert.casrel_relation_extraction_task import CasRelRETask as CasRelRETask

from weathon.nlp.model.re.casrel_bert.casrel_relation_extraction_predictor import CasRelREPredictor as Predictor
from weathon.nlp.model.re.casrel_bert.casrel_relation_extraction_predictor import CasRelREPredictor as CasRelREPredictor
