import warnings
import torch
from extractor.s3fd import SFDDetector
from extractor.utils import load_file_from_url, crop, flip, get_preds_fromhm


default_model_urls = {
    '2DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip',
    '3DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/3DFAN4-4a694010b9.zip',
    'depth': 'https://www.adrianbulat.com/downloads/python-fan/depth-6c4283c0e0.zip',
}

models_urls = {
    '1.6': {
        '2DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4_1.6-c827573f02.zip',
        '3DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/3DFAN4_1.6-ec5cf40a1d.zip',
        'depth': 'https://www.adrianbulat.com/downloads/python-fan/depth_1.6-2aa3f18772.zip',
    },
    '1.5': {
        '2DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4_1.5-a60332318a.zip',
        '3DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/3DFAN4_1.5-176570af4d.zip',
        'depth': 'https://www.adrianbulat.com/downloads/python-fan/depth_1.5-bc10f98e39.zip',
    },
}


def download_landmarks_model(network_size=4):
    network_size = int(network_size)
    pytorch_version = torch.__version__
    if 'dev' in pytorch_version:
        pytorch_version = pytorch_version.rsplit('.', 2)[0]
    else:
        pytorch_version = pytorch_version.rsplit('.', 1)[0]

    # Initialise the face alignemnt networks
    network_name = '2DFAN-' + str(network_size)
    face_alignment_net = torch.jit.load(
        load_file_from_url(models_urls.get(pytorch_version, default_model_urls)[network_name]))
    return face_alignment_net


class FaceAlignment:
    def __init__(self, path_to_detector, device='cuda', flip_input=False, verbose=False):
        self.device = device
        self.flip_input = flip_input
        self.verbose = verbose
        self.face_detector = SFDDetector()

        if 'cuda' in device:
            torch.backends.cudnn.benchmark = True

        # Initialise the face alignemnt networks
        if path_to_detector is None:
            self.face_alignment_net = download_landmarks_model()
        else:
            self.face_alignment_net = torch.jit.load(path_to_detector)

        self.face_alignment_net.to(device)
        self.face_alignment_net.eval()

    @torch.no_grad()
    def get_landmarks_from_image(self, image, detected_faces=None, return_bboxes=False,
                                 return_landmark_score=False):
        """
        Predict the landmarks for each face present in the image.
        This function predicts a set of 68 2D or 3D images, one for each image present.
        If detect_faces is None the method will also run a face detector.
         Arguments:
            image_or_path {string or numpy.array or torch.tensor} -- The input image or path to it.
        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None})
            return_bboxes {boolean} -- If True, return the face bounding boxes in addition to the keypoints.
            return_landmark_score {boolean} -- If True, return the keypoint scores along with the keypoints.
        Return:
            result:
                1. if both return_bboxes and return_landmark_score are False, result will be:
                    landmark
                2. Otherwise, result will be one of the following, depending on the actual value of return_* arguments.
                    (landmark, landmark_score, detected_face)
                    (landmark, None,           detected_face)
                    (landmark, landmark_score, None         )
        """

        if len(detected_faces) == 0:
            warnings.warn("No faces were detected.")
            if return_bboxes or return_landmark_score:
                return None, None, None
            else:
                return None

        landmarks = []
        landmarks_scores = []
        for i, d in enumerate(detected_faces):
            center = torch.tensor(
                [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
            center[1] = center[1] - (d[3] - d[1]) * 0.12
            scale = (d[2] - d[0] + d[3] - d[1]) / self.face_detector.reference_scale

            inp = crop(image, center, scale)
            inp = torch.from_numpy(inp.transpose(
                (2, 0, 1))).float()

            inp = inp.to(self.device)
            inp.div_(255.0).unsqueeze_(0)

            out = self.face_alignment_net(inp).detach()
            if self.flip_input:
                out += flip(self.face_alignment_net(flip(inp)).detach(), is_label=True)
            out = out.cpu().numpy()

            pts, pts_img, scores = get_preds_fromhm(out, center.numpy(), scale)
            pts, pts_img = torch.from_numpy(pts), torch.from_numpy(pts_img)
            pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)
            scores = scores.squeeze(0)

            landmarks.append(pts_img.numpy())
            landmarks_scores.append(scores)

        if not return_bboxes:
            detected_faces = None
        if not return_landmark_score:
            landmarks_scores = None
        if return_bboxes or return_landmark_score:
            return landmarks, landmarks_scores, detected_faces
        else:
            return landmarks
