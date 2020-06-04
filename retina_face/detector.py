import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os

from time import time
from retina_face.layers import PriorBox
from retina_face.utils import decode, decode_landm, py_cpu_nms
from retina_face.retina import RetinaFace
from torchvision.models import shufflenet_v2_x1_0


class FaceDetector:
    def __init__(self, det_threshold, checkpoint_path, device, use_apex=False, full_precision=False, top_k=750, nms_threshold=0.4):
        """
        parameters:
            config: dict (contains training/inference backbone`s parameters)
            det_threshold: float (threshold for bbox`s confidence)
            checkpoint_path: str (path to pretrained model)
            device: str (cpu or gpu inference)
            use_apex: bool (need only for not full precision and inference optimization)
            full_precision: bool ()
            top_k: int (max for detections)
            nms_threshold: int (threshold for non-maximum supression)

        """
        self.device_str = device
        self.device = torch.device(device)
        self.top_k = top_k
        self.nms_threshold = nms_threshold
        self.det_threshold = det_threshold

        torch.set_grad_enabled(False)
        model = shufflenet_v2_x1_0(pretrained=False)

        self.model = RetinaFace(model, 'inference')

        self.__load_ckpt(checkpoint_path)

        cudnn.enabled = True
        cudnn.benchmark = True

        if use_apex:
            from apex import amp
            self.model = amp.initialize(model.to(device), None, opt_level='O0' if full_precision else 'O2', keep_batchnorm_fp32=True, verbosity=0)

        self.model = self.model.to(self.device)
        self.model.eval()

        self.priorbox = None
        self.priors = None
        self.prior_data = None

        self.width, self.height = None, None
        self.scale, self.second_scale = None, None

    @staticmethod
    def __remove_prefix(state_dict, prefix):
        # print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    @staticmethod
    def __add_layers_prefix(state_dict):
        def f(layer_name):
            if layer_name.startswith('body.layers.'):
                splitted_name = layer_name.split('body.layers.')
                new_layer_name = 'body.' + splitted_name[1]
                return new_layer_name

            return layer_name

        return {f(key): value for key, value in state_dict.items()}


    @staticmethod
    def __check_keys(model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        # print("missing keys: ", len(missing_keys), "\n")
        # print("unused keys: ", len(unused_pretrained_keys), "\n")
        if len(missing_keys) == 0:
            return True
        return False

    @staticmethod
    def __add_layers_prefix(state_dict):
        def f(layer_name):
            if layer_name.startswith('body.layers.'):
                splitted_name = layer_name.split('body.layers.')
                new_layer_name = 'body.' + splitted_name[1]
                return new_layer_name

            return layer_name

        return {f(key): value for key, value in state_dict.items()}

    def __load_ckpt(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise Exception('checkpoint does not exists')

        if self.device_str == 'cpu':
            pretrained_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        else:
            pretrained_dict = torch.load(checkpoint_path, map_location=self.device_str)

        if "model_state_dict" in pretrained_dict.keys():
            state_dict = self.__remove_prefix(pretrained_dict['model_state_dict'], 'module.')
        else:
            state_dict = self.__remove_prefix(pretrained_dict, 'module.')

        if not self.__check_keys(self.model, state_dict):
            state_dict = self.__add_layers_prefix(state_dict)

        self.model.load_state_dict(state_dict, strict=False)

    def return_bboxes(self, img):
        img = np.float32(img)

        if self.width is None:
            _, self.height, self.width, _ = img.shape
            self.scale = torch.Tensor([self.width, self.height, self.width, self.height])
            self.scale = self.scale.to(self.device)

        img -= (104, 117, 123)
        img = img.transpose(0, 3, 1, 2)
        img = torch.from_numpy(img)
        img = img.to(self.device)

        batch_loc, batch_conf, batch_landms = self.model(img)

        all_dets = []
        all_landms = []

        for idx, loc in enumerate(batch_loc):
            conf = batch_conf[idx]
            landms = batch_landms[idx]
            if self.priorbox is None:
                self.priorbox = PriorBox(min_sizes=[[16, 32], [64, 128], [256, 512]], steps=[8, 16, 32], clip=False, image_size=(self.height, self.width))
                self.priors = self.priorbox.forward()
                self.priors = self.priors.to(self.device)
                self.prior_data = self.priors.data

            boxes = decode(loc.data.squeeze(0), self.prior_data, variances=[0.1, 0.1])
            try:
                boxes *= self.scale
            except TypeError:
                return list(), list()
            boxes = boxes.cpu().numpy()

            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), self.prior_data, variances=[0.1, 0.2])

            if self.second_scale is None:
                self.second_scale = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                                img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                                img.shape[3], img.shape[2]])
                self.second_scale = self.second_scale.to(self.device)

            landms = landms * self.second_scale
            landms = landms.cpu().numpy()

            inds = np.where(scores > self.det_threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            order = scores.argsort()[::-1][:self.top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, self.nms_threshold)
            dets = dets[keep, :]
            landms = landms[keep]

            dets = dets[:self.top_k, :]
            all_dets.append(dets)
            landms = landms[:self.top_k, :]
            all_landms.append(landms)

        return np.array(all_dets), np.array(all_landms)
