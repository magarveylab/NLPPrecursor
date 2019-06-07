import numpy as np
from tqdm import tqdm
import pickle
import torch

def softmax(x):
    """
    Numpy Softmax, via comments on https://gist.github.com/stober/1946926
    >>> res = softmax(np.array([0, 200, 10]))
    >>> np.sum(res)
    1.0
    >>> np.all(np.abs(res - np.array([0, 1, 0])) < 0.0001)
    True
    >>> res = softmax(np.array([[0, 200, 10], [0, 10, 200], [200, 0, 10]]))
    >>> np.sum(res, axis=1)
    array([ 1.,  1.,  1.])
    >>> res = softmax(np.array([[0, 200, 10], [0, 10, 200]]))
    >>> np.sum(res, axis=1)
    array([ 1.,  1.])
    """
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

def tokenizer(t):
    tokenized = [x for x in t.split('-')]
    tokenized.append('pad')
    return tokenized # TODO add more complex logic

def predictor (data, vocab_path, model_path, mode='CPU'):
    #Load vocab object
    with open(vocab_path, 'rb') as vocab_data:
        vocab = pickle.load(vocab_data)

    #Load model
    with open(model_path, 'rb') as model_data:
        model = pickle.load(model_data)

    #load tokenize and numericalize all inputs
    items = [vocab.numericalize(tokenizer(i['sequence'])) for i in data]

    #predict on inputs
    pred_items = []
    for ary in tqdm(items):

        ary = np.reshape(ary,(-1,1))

        # turn this array into a tensor
        tensor = torch.from_numpy(ary)

        if mode == 'GPU':
            tensor = tensor.type(torch.cuda.LongTensor)

        if mode == 'CPU':
            tensor = tensor.type(torch.LongTensor)

        # wrap in a torch Variable
        variable = torch.autograd.Variable(tensor)

        # do the predictions
        predictions = model(variable)

        # convert back to numpy
        numpy_preds = predictions[0].data.cpu().numpy()
        pred_items.append(list(softmax(numpy_preds[0])[0]))

    for x,y in zip(data, pred_items):
        x['predictions'] = {c: round(y[index],2) for c, index in vocab.class_dict.items()}

    return data
