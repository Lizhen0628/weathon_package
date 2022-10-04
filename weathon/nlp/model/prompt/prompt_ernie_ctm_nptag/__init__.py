from weathon.nlp.dataset import PromptDataset as Dataset
from weathon.nlp.dataset import PromptDataset as PromptErnieCtmNptagDataset

from weathon.nlp.model.prompt.prompt_ernie_ctm_nptag.prompt_ernie_ctm_nptag_tokenizer import \
    PromptErnieCtmNptagTokenizer
from weathon.nlp.model.prompt.prompt_ernie_ctm_nptag.prompt_ernie_ctm_nptag_tokenizer import \
    PromptErnieCtmNptagTokenizer as Tokenizer
from weathon.nlp.model.prompt.prompt_ernie_ctm_nptag.prompt_ernie_ctm_nptag_tokenizer import \
    PromptErnieCtmNptagTokenizer as PromptErnieCtmNptagTokenizer

from weathon.nlp.nn import BertConfig as PromptErnieCtmNptagConfig
from weathon.nlp.nn import BertConfig as ModuleConfig

from weathon.nlp.model.prompt.prompt_ernie_ctm_nptag.prompt_ernie_ctm_nptag import PromptErnieCtmNptag
from weathon.nlp.model.prompt.prompt_ernie_ctm_nptag.prompt_ernie_ctm_nptag import PromptErnieCtmNptag as Module

from weathon.utils.optimizer_utils import OptimizerUtils

get_default_model_optimizer = OptimizerUtils.get_default_bert_optimizer
get_default_prompt_ernie_ctm_nptag_optimizer = OptimizerUtils.get_default_bert_optimizer

from weathon.nlp.task import PromptMLMTask as Task
from weathon.nlp.task import PromptMLMTask as PromptErnieCtmNptagTask

from weathon.nlp.predictor import PromptMLMPredictor as Predictor
from weathon.nlp.predictor import PromptMLMPredictor as PromptErnieCtmNptagPredictor
