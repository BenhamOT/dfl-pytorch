import torch
import io
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url

# incase the model needs to be downloaded again
models_urls = {
        's3fd': 'https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth',
    }


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale=1.0):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.scale = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.empty(self.n_channels).fill_(self.scale))

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = x / norm * self.weight.view(1, -1, 1, 1)
        return x


class S3FD(nn.Module):
    def __init__(self):
        super(S3FD, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.fc6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=3)
        self.fc7 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)

        self.conv6_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.conv6_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.conv7_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.conv7_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv3_3_norm = L2Norm(256, scale=10)
        self.conv4_3_norm = L2Norm(512, scale=8)
        self.conv5_3_norm = L2Norm(512, scale=5)

        self.conv3_3_norm_mbox_conf = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
        self.conv3_3_norm_mbox_loc = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
        self.conv4_3_norm_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv4_3_norm_mbox_loc = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)
        self.conv5_3_norm_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv5_3_norm_mbox_loc = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)

        self.fc7_mbox_conf = nn.Conv2d(1024, 2, kernel_size=3, stride=1, padding=1)
        self.fc7_mbox_loc = nn.Conv2d(1024, 4, kernel_size=3, stride=1, padding=1)
        self.conv6_2_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv6_2_mbox_loc = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)
        self.conv7_2_mbox_conf = nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1)
        self.conv7_2_mbox_loc = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = F.relu(self.conv1_1(x), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        f3_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        h = F.relu(self.conv4_3(h), inplace=True)
        f4_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv5_1(h), inplace=True)
        h = F.relu(self.conv5_2(h), inplace=True)
        h = F.relu(self.conv5_3(h), inplace=True)
        f5_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.fc6(h), inplace=True)
        h = F.relu(self.fc7(h), inplace=True)
        ffc7 = h
        h = F.relu(self.conv6_1(h), inplace=True)
        h = F.relu(self.conv6_2(h), inplace=True)
        f6_2 = h
        h = F.relu(self.conv7_1(h), inplace=True)
        h = F.relu(self.conv7_2(h), inplace=True)
        f7_2 = h

        f3_3 = self.conv3_3_norm(f3_3)
        f4_3 = self.conv4_3_norm(f4_3)
        f5_3 = self.conv5_3_norm(f5_3)

        cls1 = self.conv3_3_norm_mbox_conf(f3_3)
        reg1 = self.conv3_3_norm_mbox_loc(f3_3)
        cls2 = self.conv4_3_norm_mbox_conf(f4_3)
        reg2 = self.conv4_3_norm_mbox_loc(f4_3)
        cls3 = self.conv5_3_norm_mbox_conf(f5_3)
        reg3 = self.conv5_3_norm_mbox_loc(f5_3)
        cls4 = self.fc7_mbox_conf(ffc7)
        reg4 = self.fc7_mbox_loc(ffc7)
        cls5 = self.conv6_2_mbox_conf(f6_2)
        reg5 = self.conv6_2_mbox_loc(f6_2)
        cls6 = self.conv7_2_mbox_conf(f7_2)
        reg6 = self.conv7_2_mbox_loc(f7_2)

        # max-out background label
        chunk = torch.chunk(cls1, 4, 1)
        bmax = torch.max(torch.max(chunk[0], chunk[1]), chunk[2])
        cls1 = torch.cat([bmax, chunk[3]], dim=1)

        return [cls1, reg1, cls2, reg2, cls3, reg3, cls4, reg4, cls5, reg5, cls6, reg6]


class SFDDetector:
    def __init__(self, device="cpu", path_to_detector=None, verbose=False, filter_threshold=0.5):

        # Initialise the face detector
        if path_to_detector is None:
            model_weights = load_url(models_urls['s3fd'])
        else:
            model_weights = torch.load(path_to_detector)

        self.device = device
        self.fiter_threshold = filter_threshold
        self.face_detector = S3FD()
        self.face_detector.load_state_dict(model_weights)
        self.face_detector.to(device)
        self.face_detector.eval()

    def _filter_bboxes(self, bboxlist):
        if len(bboxlist) > 0:
            keep = self.nms(bboxlist, 0.3)
            bboxlist = bboxlist[keep, :]
            bboxlist = [x for x in bboxlist if x[-1] > self.fiter_threshold]

        return bboxlist

    def detect_from_image(self, tensor_or_path):
        image = self.tensor_or_path_to_ndarray(tensor_or_path)

        bboxlist = self.detect(self.face_detector, image, device=self.device)[0]
        bboxlist = self._filter_bboxes(bboxlist)
        bboxlist = self.remove_low_pixel_imgs(bounding_box_list=bboxlist)
        return bboxlist

    @staticmethod
    def remove_low_pixel_imgs(bounding_box_list):

        for counter, bb_array in enumerate(bounding_box_list):
            left, top, right, bottom, _ = bb_array
            if min(abs(left - right), abs(top - bottom)) < 40:
                bounding_box_list.pop(counter)

        return bounding_box_list

    @staticmethod
    def tensor_or_path_to_ndarray(tensor_or_path):
        """
        Convert path (represented as a string) or torch.tensor to a numpy.ndarray
        Arguments:
            tensor_or_path {numpy.ndarray, torch.tensor or string} -- path to the image, or the image itself
        """
        if isinstance(tensor_or_path, str):
            return io.imread(tensor_or_path)
        elif torch.is_tensor(tensor_or_path):
            return tensor_or_path.cpu().numpy()
        elif isinstance(tensor_or_path, np.ndarray):
            return tensor_or_path
        else:
            raise TypeError

    def detect(self, net, img, device):
        img = img.transpose(2, 0, 1)
        # Creates a batch of 1
        img = np.expand_dims(img, 0)

        img = torch.from_numpy(img.copy()).to(device, dtype=torch.float32)

        return self.batch_detect(net, img, device)

    def batch_detect(self, net, img_batch, device):
        """
        Inputs:
            - img_batch: a torch.Tensor of shape (Batch size, Channels, Height, Width)
        """
        if 'cuda' in device:
            torch.backends.cudnn.benchmark = True

        batch_size = img_batch.size(0)
        img_batch = img_batch.to(device, dtype=torch.float32)

        img_batch = img_batch.flip(-3)  # RGB to BGR
        img_batch = img_batch - torch.tensor([104.0, 117.0, 123.0], device=device).view(1, 3, 1, 1)

        with torch.no_grad():
            olist = net(img_batch)  # patched uint8_t overflow error

        for i in range(len(olist) // 2):
            olist[i * 2] = F.softmax(olist[i * 2], dim=1)

        olist = [oelem.data.cpu().numpy() for oelem in olist]

        bboxlists = self.get_predictions(olist, batch_size)
        return bboxlists

    def get_predictions(self, olist, batch_size):
        bboxlists = []
        variances = [0.1, 0.2]
        for j in range(batch_size):
            bboxlist = []
            for i in range(len(olist) // 2):
                ocls, oreg = olist[i * 2], olist[i * 2 + 1]
                stride = 2 ** (i + 2)  # 4,8,16,32,64,128
                poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
                for Iindex, hindex, windex in poss:
                    axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
                    score = ocls[j, 1, hindex, windex]
                    loc = oreg[j, :, hindex, windex].copy().reshape(1, 4)
                    priors = np.array([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]])
                    box = self.decode(loc, priors, variances)
                    x1, y1, x2, y2 = box[0]
                    bboxlist.append([x1, y1, x2, y2, score])

            bboxlists.append(bboxlist)

        bboxlists = np.array(bboxlists)
        return bboxlists

    def flip_detect(self, net, img, device):
        img = cv2.flip(img, 1)
        b = self.detect(net, img, device)

        bboxlist = np.zeros(b.shape)
        bboxlist[:, 0] = img.shape[1] - b[:, 2]
        bboxlist[:, 1] = b[:, 1]
        bboxlist[:, 2] = img.shape[1] - b[:, 0]
        bboxlist[:, 3] = b[:, 3]
        bboxlist[:, 4] = b[:, 4]
        return bboxlist

    @staticmethod
    def pts_to_bb(pts):
        min_x, min_y = np.min(pts, axis=0)
        max_x, max_y = np.max(pts, axis=0)
        return np.array([min_x, min_y, max_x, max_y])

    @staticmethod
    def nms(dets, thresh):
        if 0 == len(dets):
            return []
        x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
            xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])

            w, h = np.maximum(0.0, xx2 - xx1 + 1), np.maximum(0.0, yy2 - yy1 + 1)
            ovr = w * h / (areas[i] + areas[order[1:]] - w * h)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    @staticmethod
    def encode(matched, priors, variances):
        """Encode the variances from the priorbox layers into the ground truth boxes
        we have matched (based on jaccard overlap) with the prior boxes.
        Args:
            matched: (tensor) Coords of ground truth for each prior in point-form
                Shape: [num_priors, 4].
            priors: (tensor) Prior boxes in center-offset form
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            encoded boxes (tensor), Shape: [num_priors, 4]
        """
        # dist b/t match center and prior's center
        g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
        # encode variance
        g_cxcy /= (variances[0] * priors[:, 2:])
        # match wh / prior wh
        g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
        g_wh = np.log(g_wh) / variances[1]

        # return target for smooth_l1_loss
        return np.concatenate([g_cxcy, g_wh], 1)  # [num_priors,4]

    @staticmethod
    def decode(loc, priors, variances):
        """Decode locations from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            loc (tensor): location predictions for loc layers,
                Shape: [num_priors,4]
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            decoded bounding box predictions
        """
        boxes = np.concatenate((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0