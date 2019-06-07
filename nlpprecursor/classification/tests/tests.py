
'''
This is deprecated and better described in README.md
'''

from nlpprecursor.classification.data import DatasetGenerator
from pathlib import Path
import json


def train_opt(lm_json, class_json, data_path):
    dg = DatasetGenerator(0.9, lm_json, class_json, data_path, bs=2)
    print(dg.stage)
    dg._read_jsons()
    dg.stage += 1
    print(dg.stage)

    dg.tokenize()
    dg.stage += 1
    print(dg.stage)

    dg.split_class_data()
    dg.stage += 1
    print(dg.stage)

    dg.train_lm(epochs=1)
    dg.stage +=1
    print(dg.stage)

    dg.train_class(epochs=1)

    dg.test_class()


def test_model():
    model_path = "/home/nmerwin/adapsyn/projects/ulmfit/prot_fastai/test_resources/classification/new_dataset/models/final_model.pkl"
    sequences = [{"sequence":"MKADQEQKLPPPPLQIVSTCII", "name":"test"}]
    vocab_path = "/home/nmerwin/adapsyn/projects/ulmfit/prot_fastai/test_resources/classification/new_dataset/vocab.pkl"
    predictions = DatasetGenerator.predict(model_path, vocab_path, sequences)
    print(json.dumps(predictions))


if __name__ == "__main__":
    data_path = Path("/home/nmerwin/adapsyn/projects/ulmfit/prot_fastai/test_resources/classification/small_dataset")
    lm_json = data_path / "lm_data.json"
    class_json = data_path / "class_data.json"
    train_opt(lm_json, class_json, data_path)
    test_model()
