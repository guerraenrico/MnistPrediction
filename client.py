from __future__ import print_function

import argparse
import time
import numpy as np
# from scipy.misc import imread
from imageio import imread
import grpc
from tensorflow.contrib.util import make_tensor_proto

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import classification_pb2

def run(host, port, image, model, signature_name):
    channel = grpc.insecure_channel('{host}:{port}'.format(host=host, port=port))
    # channel = grpc.insecure_channel('https://mnist-prediction.herokuapp.com/')

    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    data = imread(image)
    data = data.astype(np.float32)
    print('data', data)

    start = time.time()

    request = predict_pb2.PredictRequest()
    request.model_spec.name = model
    request.model_spec.signature_name = signature_name
    request.inputs['pixels'].CopyFrom(make_tensor_proto(data, shape=(1, 280, 280, 1)))

    result = stub.Predict(request, 10.0)

    end = time.time()
    time_diff = end - start
    print('result', result)
    print('time elapsed {}'.format(time_diff))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', help='Tensorflow server host name', default='192.168.99.100')
    parser.add_argument('--port', help='Tensorflow server port number', default=8500, type=int)
    parser.add_argument('--image', help='input image', default='./data/2.png', type=str)
    parser.add_argument('--model', help='model name', type=str, default="mnist_prediction")
    parser.add_argument('--signature_name', help='Signature name of saved TF model',
                        default='predict', type=str)

    args = parser.parse_args()
    run(args.host, args.port, args.image, args.model, args.signature_name)
