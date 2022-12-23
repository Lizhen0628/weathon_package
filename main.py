from pathlib import Path
from weathon.nlp.dataset import BIONERDataset as Dataset
from weathon.nlp.processor.tokenizer import TokenTokenizer as Tokenizer
from weathon.nlp.model.ner.crf_bert import CrfBert as Model
from weathon.nlp.task.named_entity_recognition import CRFNERTask as Task
from weathon.utils import OptimizerUtils
from transformers import AutoConfig


def main(transformer_model):
    train_file = Path("weathon/data/cluener/train.jsonl")
    dev_file = Path("weathon/data/cluener/dev.jsonl")
    test_file = Path("weathon/data/cluener/test.json")
    # dataset
    ner_train_dataset = Dataset(train_file)
    ner_dev_dataset = Dataset(dev_file)

    # tokenizer
    tokenizer = Tokenizer(transformer_model, max_seq_len=50)
    # 文本切分、ID化
    ner_train_dataset.convert_to_ids(tokenizer)
    ner_dev_dataset.convert_to_ids(tokenizer)

    # 加载预训练模型
    config = AutoConfig.from_pretrained(transformer_model, num_labels=len(ner_train_dataset.cat2id))
    dl_module = Model.from_pretrained(transformer_model, config=config)

    optimizer = OptimizerUtils.get_default_optimizer(dl_module, "crf_bert", lr=1e-3, crf_lr=2e-3)

    task = Task(dl_module, optimizer, 'ce')

    task.fit(ner_train_dataset, ner_dev_dataset, lr=2e-3, epochs=2, batch_size=16)

    print(dl_module)


if __name__ == '__main__':
    main(transformer_model="clue/albert_chinese_tiny")
