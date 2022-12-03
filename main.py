from pathlib import Path
from weathon.nlp.dataset import BIONERDataset as Dataset

def main():

    train_file = Path("weathon/data/cluener/train.json")
    dev_file = Path("weathon/data/cluener/dev.json")
    test_file = Path("weathon/data/cluener/test.json")
    # dataset
    ner_train_dataset = Dataset(train_file)
    ner_dev_dataset = Dataset(dev_file)




if __name__ == '__main__':
    main()