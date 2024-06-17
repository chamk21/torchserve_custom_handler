# custom handler file

# model_handler.py

"""
ModelHandler defines a custom model handler.
"""
import torch
import logging
import os
from ts.torch_handler.base_handler import BaseHandler
import ast
import torch
import torch.nn as nn
import torch.optim as optim


class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0

    def initialize(self, context):
        #  load the model
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(" device is selected")
        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")
       
        self.model = torch.jit.load(model_pt_path)
        self.model.eval()
        self.initialized = True

    def preprocess(self, data):
        preprocessed_data = data[0].get("data")
        if preprocessed_data is None:
            preprocessed_data = data[0].get("body")
            dict_str = preprocessed_data.decode("UTF-8")
            actual_data = ast.literal_eval(dict_str)
            for key,value in actual_data.items():
                data_array = value
                data_dictionary_inside_data_array = data_array[0]
                for key,value in data_dictionary_inside_data_array.items():
                    final_output_tensor = torch.FloatTensor(value)
        return final_output_tensor


    def inference(self, model_input):
        model_output = self.model.forward(model_input)
        return model_output

    def postprocess(self, inference_output):
        # Take output from network and post-process to desired format
        postprocess_output = inference_output
        np_arr = postprocess_output.detach().cpu().numpy()
        np_arr_list = np_arr.tolist()
        return np_arr_list

    def handle(self, data, context):
        model_input = self.preprocess(data)
        logging.info("===============================INPUT_DATA================================")
        logging.info(model_input)
        logging.info(type(model_input))
        logging.info("=========================================================================")
        model_output = self.inference(model_input)
        logging.info("===============================OUTPUT_DATA_AFTER_MODEL.FORWARD================================")
        logging.info(model_output)
        logging.info("=========================================================================")
        return self.postprocess(model_output)
