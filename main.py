from pathlib import Path
from weathon.nlp.dataset import BIONERDataset as Dataset
from weathon.nlp.processor.tokenizer import TokenTokenizer as Tokenizer
from weathon.nlp.nn.configuration import BertConfig as Config


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
    config = Config.from_pretrained(transformer_model, num_labels=len(ner_train_dataset.cat2id))
    dl_module = CRFBert.from_pretrained(transformer_model, config=config)

if __name__ == '__main__':
    transformer_model = "clue/albert_chinese_tiny"
    main(transformer_model)
