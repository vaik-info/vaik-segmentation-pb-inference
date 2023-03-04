from typing import List, Dict, Tuple
import tensorflow as tf
from PIL import Image
import numpy as np

class PbModel:
    def __init__(self, input_saved_model_dir_path: str = None, classes: Tuple = None):
        self.model = tf.saved_model.load(input_saved_model_dir_path)
        self.model_input_shape = self.model.signatures["serving_default"].inputs[0].shape
        self.model_input_dtype = self.model.signatures["serving_default"].inputs[0].dtype
        self.model_output_shape = self.model.signatures["serving_default"].outputs[0].shape
        self.model_output_dtype = self.model.signatures["serving_default"].outputs[0].dtype
        self.classes = classes

    def inference(self, input_image_list: List[np.ndarray], batch_size: int = 8) -> Tuple[List[Dict], np.array]:
        resized_image_array, resize_image_shape_list, input_image_shape_list = self.__preprocess_image_list(input_image_list, self.model_input_shape[1:3])
        raw_pred = self.__inference(resized_image_array, batch_size)
        output = self.__output_parse(raw_pred, resize_image_shape_list, input_image_shape_list)
        return output, raw_pred

    def __inference(self, resize_input_tensor: np.ndarray, batch_size: int) -> np.ndarray:
        if len(resize_input_tensor.shape) != 4:
            raise ValueError('dimension mismatch')
        if not np.issubdtype(resize_input_tensor.dtype, np.uint8):
            raise ValueError(f'dtype mismatch expected: {np.uint8}, actual: {resize_input_tensor.dtype}')

        output_tensor = tf.zeros((resize_input_tensor.shape[0], )+ self.model_output_shape[1:],
                                 self.model_output_dtype).numpy()
        for index in range(0, resize_input_tensor.shape[0], batch_size):
            batch = resize_input_tensor[index:index + batch_size, :, :, :]
            batch_pad = tf.zeros(((batch_size, ) + self.model_input_shape[1:]), self.model_input_dtype).numpy()
            batch_pad[:batch.shape[0], :, :, :] = batch
            raw_pred = self.model(batch_pad)
            output_tensor[index:index + batch.shape[0]] = raw_pred[:batch.shape[0]]
        return output_tensor

    def __preprocess_image_list(self, input_image_list: List[np.ndarray],
                                resize_input_shape: Tuple[int, int]) -> Tuple[np.ndarray, List[Tuple], List[Tuple]]:
        resized_image_list, resize_image_shape_list, input_image_shape_list = [], [], []
        for input_image in input_image_list:
            resized_image, resize_image_shape, input_image_shape = self.__preprocess_image(input_image, resize_input_shape)
            resized_image_list.append(resized_image)
            resize_image_shape_list.append(resize_image_shape)
            input_image_shape_list.append(input_image_shape)
        return np.stack(resized_image_list), resize_image_shape_list, input_image_shape_list

    def __preprocess_image(self, input_image: np.ndarray, resize_input_shape: Tuple[int, int]) -> Tuple[
        np.ndarray, Tuple[int, int], Tuple[int, int]]:
        if len(input_image.shape) != 3:
            raise ValueError('dimension mismatch')
        if not np.issubdtype(input_image.dtype, np.uint8):
            raise ValueError(f'dtype mismatch expected: {np.uint8}, actual: {input_image.dtype}')

        output_image = np.zeros((*resize_input_shape, input_image.shape[2]),
                                dtype=input_image.dtype)
        pil_image = Image.fromarray(input_image)
        x_ratio, y_ratio = resize_input_shape[1] / pil_image.width, resize_input_shape[0] / pil_image.height
        if x_ratio < y_ratio:
            resize_size = (resize_input_shape[1], round(pil_image.height * x_ratio))
        else:
            resize_size = (round(pil_image.width * y_ratio), resize_input_shape[0])
        resize_pil_image = pil_image.resize(resize_size)
        resize_image = np.array(resize_pil_image)
        output_image[:resize_image.shape[0], :resize_image.shape[1], :] = resize_image
        return output_image, resize_image.shape, input_image.shape

    def __output_parse(self, pred: np.ndarray, resize_image_shape_list: List[Tuple], input_image_shape_list: List[Tuple]) -> List[Dict]:
        output_dict_list = []
        for index in range(pred.shape[0]):
            arg_max_pred = np.argmax(pred[index], -1)
            arg_max_pred = arg_max_pred[:resize_image_shape_list[index][0], :resize_image_shape_list[index][1]]
            arg_max_pred_image = Image.fromarray(arg_max_pred.astype(np.uint8))
            arg_max_pred_image = arg_max_pred_image.resize((input_image_shape_list[index][1], input_image_shape_list[index][0]), resample=Image.BILINEAR)
            arg_max_array = np.asarray(arg_max_pred_image)
            output_dict = {'labels': arg_max_array}
            output_dict_list.append(output_dict)
        return output_dict_list
