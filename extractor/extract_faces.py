import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from extractor.utils import pil_loader, mkdir_or_delete_existing_files
from extractor.s3fd import SFDDetector
from extractor.fan2d import FaceAlignment
from extractor.landmarks_processor import get_transform_mat


class Data:
    def __init__(self, filepath=None, rects=None, landmarks=None, final_output_files=None):
        self.filepath = filepath
        self.file_name = filepath.split("/")[-1].replace(".jpg", "").replace(".png", "")
        self.rects = rects or []
        self.rects_rotation = 0
        self.landmarks = landmarks or []
        self.final_output_files = final_output_files or []
        self.faces_detected = 0


class ExtractFaces:
    def __init__(self, input_data, image_size=None, jpeg_quality=None, max_faces_from_image=0,
                 images_output_path=None, landmarks_output_path=None, device_config=None):

        self.input_data = input_data
        self.image_size = image_size
        self.jpg_quality = jpeg_quality
        self.max_faces_from_image = max_faces_from_image
        self.images_output_path = images_output_path
        self.landmarks_output_path = landmarks_output_path
        self.device_config = device_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)
        self.rects_extractor = SFDDetector(device=self.device, path_to_detector="extractor/models/s3fd.pth")
        self.landmarks_extractor = FaceAlignment(landmarks_type=1, device=self.device)
        self.detected_faces = None

    def run(self):
        self.detected_faces = 0
        for image_file_path in tqdm(self.input_data):
            data = Data(filepath=image_file_path)
            data = self.process_data(data)
            self.detected_faces += data.faces_detected

    def number_of_detected_faces(self):
        if self.detected_faces is None:
            print("need to call run method")
        return self.detected_faces

    def process_data(self, data):
        filepath = data.filepath
        image = pil_loader(filepath)

        #
        data = self.rects_stage(
            data=data,
            image=image.copy(),
            max_faces_from_image=self.max_faces_from_image,
            rects_extractor=self.rects_extractor
        )
        data = self.landmarks_stage(
            data=data,
            image=image.copy(),
            landmarks_extractor=self.landmarks_extractor
        )
        data = self.final_stage(
            data=data,
            image=image,
            image_size=self.image_size
        )
        return data

    @staticmethod
    def rects_stage(data, image, max_faces_from_image, rects_extractor):
        h,w,c = image.shape
        if min(h,w) < 128:
            # Image is too small
            data.rects = []
        else:
            for rot in ([0, 90, 270, 180]):
                if rot == 0:
                    rotated_image = image
                elif rot == 90:
                    rotated_image = image.swapaxes( 0,1 )[:,::-1,:]
                elif rot == 180:
                    rotated_image = image[::-1,::-1,:]
                elif rot == 270:
                    rotated_image = image.swapaxes( 0,1 )[::-1,:,:]
                rects = data.rects = rects_extractor.detect_from_image(rotated_image)
                if len(rects) != 0:
                    data.rects_rotation = rot
                    break
            if max_faces_from_image is not None and \
               max_faces_from_image > 0 and \
               len(data.rects) > 0:
                data.rects = data.rects[0:max_faces_from_image]

        return data

    @staticmethod
    def landmarks_stage(data, image, landmarks_extractor):
        if not data.rects:
            return data

        h, w, ch = image.shape

        if data.rects_rotation == 0:
            rotated_image = image
        elif data.rects_rotation == 90:
            rotated_image = image.swapaxes( 0,1 )[:,::-1,:]
        elif data.rects_rotation == 180:
            rotated_image = image[::-1,::-1,:]
        elif data.rects_rotation == 270:
            rotated_image = image.swapaxes( 0,1 )[::-1,:,:]

        data.landmarks = landmarks_extractor.get_landmarks_from_image(rotated_image, data.rects)
        if data.rects_rotation != 0:
            for i, (rect, lmrks) in enumerate(zip(data.rects, data.landmarks)):
                new_rect, new_lmrks = rect, lmrks
                (l,t,r,b, _) = rect
                if data.rects_rotation == 90:
                    new_rect = ( t, h-l, b, h-r)
                    if lmrks is not None:
                        new_lmrks = lmrks[:,::-1].copy()
                        new_lmrks[:,1] = h - new_lmrks[:,1]
                elif data.rects_rotation == 180:
                    if lmrks is not None:
                        new_rect = ( w-l, h-t, w-r, h-b)
                        new_lmrks = lmrks.copy()
                        new_lmrks[:,0] = w - new_lmrks[:,0]
                        new_lmrks[:,1] = h - new_lmrks[:,1]
                elif data.rects_rotation == 270:
                    new_rect = ( w-b, l, w-t, r )
                    if lmrks is not None:
                        new_lmrks = lmrks[:,::-1].copy()
                        new_lmrks[:,0] = w - new_lmrks[:,0]
                data.rects[i], data.landmarks[i] = new_rect, new_lmrks

        return data

    def final_stage(self, data, image, image_size):
        data.final_output_files = []
        file_name = data.file_name
        rects = data.rects
        landmarks = data.landmarks

        if landmarks is None:
            return data

        face_idx = 0
        for rect, image_landmarks in zip(rects, landmarks):

            image_to_face_mat = get_transform_mat(image_landmarks, image_size)
            face_image = cv2.warpAffine(image, image_to_face_mat, (image_size, image_size), cv2.INTER_LANCZOS4)
            face_image = Image.fromarray(face_image)
            # save the image
            images_output_filepath = self.images_output_path + f"{file_name}_{face_idx}.jpg"
            face_image.save(images_output_filepath)
            # save the landmakrs
            landmarks_output_filepath = self.landmarks_output_path + f"{file_name}_{face_idx}.npy"
            np.save(landmarks_output_filepath, image_landmarks)

            data.final_output_files.append(images_output_filepath)
            face_idx += 1

        data.faces_detected = face_idx
        return data


def extract_faces_from_frames(
        input_path,
        images_output_path=None,
        landmarks_output_path=None,
        max_faces_from_image=None,
        image_size=None,
        jpeg_quality=None,
        ):

    input_image_paths = [os.path.join(input_path, x) for x in os.listdir(input_path) if x.endswith((".jpg", ".png"))]

    # delete files from aligned or landmarks dir if it's not empty
    mkdir_or_delete_existing_files(path=images_output_path)
    mkdir_or_delete_existing_files(path=landmarks_output_path)

    images_found = len(input_image_paths)
    faces_detected = 0
    if images_found != 0:
        print('Extracting faces...')
        ExtractFaces(
            input_image_paths,
            image_size,
            jpeg_quality,
            max_faces_from_image=max_faces_from_image,
            images_output_path=images_output_path,
            landmarks_output_path=landmarks_output_path
        ).run()

    # faces_detected += sum([d.faces_detected for d in data])

    # print('-------------------------')
    # print('Images found:        %d' % (images_found) )
    # print('Faces detected:      %d' % (faces_detected) )
    # print('-------------------------')
