import json
from pathlib import Path


# train_data_df的columns必选包含"text"和"label"
# text列为文本
# label列为列表形式，列表中每个元素是如下组织的字典
# {'start_idx': 实体首字符在文本的位置, 'end_idx': 实体尾字符在文本的位置, 'type': 实体类型标签, 'entity': 实体}

def convert_label(file_path: Path, out_file: Path):
    with file_path.open('r', encoding='utf8') as reader, out_file.open('w', encoding='utf8') as writer:
        for line in reader:
            json_line = json.loads(line)
            labels = []
            for type_name, entity_dict in json_line['label'].items():
                for entity_name, entity_pos in entity_dict.items():
                    for (start_idx,end_idx) in entity_pos:
                        labels.append({
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'type': type_name,
                            'entity': entity_name
                        })

            writer.write(json.dumps({
                "text":json_line["text"],
                "label":labels
            },ensure_ascii=False) + "\n")


if __name__ == '__main__':
    train_path = Path("./train.json")
    train_file = Path("./train.jsonl")
    convert_label(train_path,train_file)

    dev_path = Path("./dev.json")
    dev_file = Path("./dev.jsonl")
    convert_label(dev_path,dev_file)
