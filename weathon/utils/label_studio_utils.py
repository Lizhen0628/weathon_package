# -*- coding: utf-8 -*-
# @Time    : 2022/10/2 17:23
# @Author  : LiZhen
# @FileName: label_studio_utils.py
# @github  : https://github.com/Lizhen0628
# @Description:


import json
import random
import time
import string
from pathlib import Path


class LabelStudioUtils:

    @staticmethod
    def jsonl2labelstudio_ner(input_file: Path, out_file: Path):
        """
        task: 命名实体识别
            将jsonl格式的数据 转化成 label studio标注结果格式
            jsonl 数据格式
            {
            "id": "train_4529",
            "text": "众兴水库位于肥东县城北15千米，东经117度26秒，北纬32度01秒，店埠河上游，属巢湖水系南淝河，是淠史杭灌区滁河干渠上的，控制流域面积114平方千米，总库容9948万立方米。",
            "labels": [[56, 58, "RIV", "滁河"], [46, 49, "RIV", "南淝河"], [42, 44, "LAK", "巢湖"], [0, 4, "RES", "众兴水库"], [6, 9, "LOC", "肥东县"], [35, 38, "RIV", "店埠河"], [65, 69, "TER", "流域面积"]]
            }
        :param input_file:  jsonl 格式文本文件
        :param out_file: label studio 结果格式：
            [
              {
                "completions": [
                  {
                    "created_at": 1611800785,
                    "id": 2001,
                    "lead_time": 65.289,
                    "result": [
                      {
                        "from_name": "label",
                        "id": "8KPMNKSZTo",
                        "to_name": "text",
                        "type": "labels",
                        "value": {
                          "end": 18,
                          "labels": [
                            "person"
                          ],
                          "start": 13,
                          "text": "1700年"
                        }
                      }
                    ]
                  }
                ],
                "data": {
                  "text": "\"巢湖的人工围垦可以追溯到1700年以前，原来的湖泊有360多个天然湖汊、湖湾都被历代所围垦，仅清代沿湖围垦总面积就达到62.8万亩。\""
                },
                "id": 2
              }

            ]
        :return:
        """
        out_writer = out_file.open('w', encoding='utf8')
        all_data = []
        label_type = set()
        with input_file.open("r", encoding="utf8") as f:
            for idx, line in enumerate(f):
                # json_line = json.loads(line)
                json_line = eval(line)
                results = []
                for idx, label in enumerate(json_line['label']):
                    label_type.add(label[2])
                    results.append(
                        {
                            "from_name": "label",
                            "id": ''.join(random.sample(string.ascii_letters + string.digits, 10)),
                            "to_name": "text",
                            "type": "labels",
                            "value": {
                                "start": int(label[0]),
                                "end": int(label[1]),
                                "text": json_line['text'],
                                "labels": [label[2]]
                            }
                        }
                    )

                all_data.append({
                    "id": idx + 1,
                    "data": {
                        "text": json_line['text']
                    },
                    "annotations": [
                        {
                            "created_at": int(time.time()),
                            "id": 1,
                            "lead_time": 1,
                            "result": results
                        }
                    ],
                    "predictions": [],
                    "meta": {},
                    "created_at": '{}T{}Z'.format(time.strftime("%Y-%m-%d", time.localtime()),
                                                  time.strftime("%H:%M:%S")),
                    "updated_at": '{}T{}Z'.format(time.strftime("%Y-%m-%d", time.localtime()),
                                                  time.strftime("%H:%M:%S")),
                })

        json.dump(all_data, out_writer, ensure_ascii=False)

        out_writer.close()
        print(f"dataset label is :{label_type}")
        print("""
        <View>
            <Labels name="label" toName="text">
        """)
        for label in label_type:
            color_str = "#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
            print(f"\t\t\t<Label value=\"{label}\" background=\"{color_str}\"/>")

        print("""
                </Labels>
            <Text name="text" value="$text"/>
        </View>
        """)

    @staticmethod
    def labelstudioner2jsonl(input_file: Path, out_file: Path):
        """
        label studio 标注结果转换成 jsonl格式：
        {
            "id": "train_4529",
            "text": "众兴水库位于肥东县城北15千米，东经117度26秒，北纬32度01秒，店埠河上游，属巢湖水系南淝河，是淠史杭灌区滁河干渠上的，控制流域面积114平方千米，总库容9948万立方米。",
            "labels": [[56, 58, "RIV", "滁河"], [46, 49, "RIV", "南淝河"], [42, 44, "LAK", "巢湖"], [0, 4, "RES", "众兴水库"], [6, 9, "LOC", "肥东县"], [35, 38, "RIV", "店埠河"], [65, 69, "TER", "流域面积"]]
            }

        :param input_file: label studio 标注格式文件路径
        :param out_file: jsonl 格式文件路径
        :return:
        """
        out_writer = Path(out_file).open("w", encoding="utf8")
        with Path(input_file).open("r", encoding="utf8") as f:
            for idx, item in enumerate(json.load(f)):
                labels = []
                if 'label' in item:
                    for label in item["label"]:
                        labels.append([int(label['start']), int(label['end']), label['labels'][0], label['text']])
                out_writer.write(json.dumps({
                    "id": idx,
                    "text": item['text'],
                    "labels": labels
                }, ensure_ascii=False) + '\n')
        out_writer.close()


if __name__ == '__main__':
    lsu = LabelStudioUtils()

    lsu.jsonl2labelstudio_ner(input_file=Path('/Users/lizhen/Downloads/train_500.jsonl'),
                              out_file=Path('/Users/lizhen/Downloads/train_500_label_studio.jsonl'))
