from pathlib import Path
from weathon.nlp.dataset import BIONERDataset as Dataset
from weathon.nlp.processor.tokenizer import TokenTokenizer as Tokenizer
from weathon.nlp.model.ner.crf_bert import CrfBert as Model


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
    dl_module = Model().from_pretrained()

    print(dl_module)

if __name__ == '__main__':
    transformer_model = "clue/albert_chinese_tiny"
    main(transformer_model)
