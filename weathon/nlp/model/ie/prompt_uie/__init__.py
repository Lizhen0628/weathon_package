from weathon.nlp.model.ie.prompt_uie.prompt_uie_information_extraction_dataset import PromptUIEDataset
from weathon.nlp.model.ie.prompt_uie.prompt_uie_information_extraction_dataset import PromptUIEDataset as Dataset

from weathon.nlp.processor.tokenizer import TransfomerTokenizer as Tokenizer
from weathon.nlp.processor.tokenizer import TransfomerTokenizer as PromptUIETokenizer

from weathon.nlp.nn import BertConfig as PromptUIEConfig
from weathon.nlp.nn import BertConfig as ModuleConfig

from weathon.nlp.model.ie.prompt_uie.prompt_uie import PromptUIE
from weathon.nlp.model.ie.prompt_uie.prompt_uie import PromptUIE as Module

from weathon.utils.optimizer_utils import OptimizerUtils
get_default_model_optimizer = OptimizerUtils.get_default_bert_optimizer
get_default_prompt_uie_optimizer = OptimizerUtils.get_default_bert_optimizer

from weathon.nlp.model.ie.prompt_uie.prompt_uie_information_extraction_task import PromptUIETask
from weathon.nlp.model.ie.prompt_uie.prompt_uie_information_extraction_task import PromptUIETask as Task

from weathon.nlp.model.ie.prompt_uie.prompt_uie_information_extraction_predictor import PromptUIEPredictor
from weathon.nlp.model.ie.prompt_uie.prompt_uie_information_extraction_predictor import PromptUIEPredictor as Predictor

