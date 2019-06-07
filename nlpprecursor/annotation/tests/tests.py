
'''
This is deprecated and better described in README.md
'''

from nlpprecursor.annotation.data import DatasetGenerator
import pickle
import json
from pathlib import Path



def train(data_path, json_path):
    dg = DatasetGenerator(0.9, json_path, data_path, bs=5)
    dg.run(1)


def test(data_path, raw_data_path):

    data_path = Path(data_path)
    model_path = data_path / "model.p"
    vocab_path = data_path / "vocab.pkl"
    datasplit_path = data_path / "datasplit.json"

    results = DatasetGenerator.evaluate_later(model_path, vocab_path, datasplit_path, raw_data_path)
    outpath = Path(data_path) / "tested.json"
    with outpath.open("w") as fp:
        json.dump(results, fp)

def predict(data_path, sequences):
    data_path = Path(data_path)
    model_path = data_path / "model.p"
    vocab_path = data_path / "vocab.pkl"




if __name__ == "__main__":
    data_path = "/home/nmerwin/adapsyn/projects/ulmfit/prot_fastai/test_resources/annotation/"
    json_path = "/home/nmerwin/adapsyn/projects/ulmfit/prot_fastai/test_resources/annotation/sampled_props.json"

    data_path = "/home/nmerwin/adapsyn/projects/barley/cleavage_models/testing_gpu_cpu/data"
    json_path = "/home/nmerwin/adapsyn/projects/barley/cleavage_models/testing_gpu_cpu/data/sampled_props.json"

    #train(data_path, json_path)
    test(data_path, json_path)
