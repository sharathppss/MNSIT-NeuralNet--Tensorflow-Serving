import json

layer1 = {"type": "conv", "filters": 32, "kernal": [5, 5], "padding": "same","activation": "relu"}
layer2 = {"type": "polling", "pool_size": [2, 2], "strides": 2, "activation": "relu"}
layer3 = {"type": "conv", "filters": 64, "kernal": [5, 5], "padding": "same","activation": "relu"}
layer4 = {"type": "polling", "pool_size": [2, 2], "strides": 2, "activation": "relu"}
layer5 = {"type": "dense", "units": 1024, "activation": "relu"}
layer6 = {"type": "dropout", "rate": 0.4 }
layer7 = {"type": "dense", "units": 10, "activation": None}


config = {  "epochs": 1,
            "learning_rate": 20,

            "loss_functions": "softmax",
            "eval_function": "accuracy",
            "optimiser": "AdamOptimizer" }


with open('NNBasicArch.config', 'w') as fp:
    json.dump(config, fp)
