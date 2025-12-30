import argparse
from typing import Union

from flask import Flask, jsonify, request
from flask_cors import CORS
import PIL.Image
import io
import os

#find the data directory
cur_dir = os.path.dirname(os.path.realpath(__file__))

print ("Current directory: {}".format(cur_dir))

data_dir = os.path.normpath(os.path.join(cur_dir, "..", "data"))
if not (os.path.exists(data_dir) and os.path.isdir(data_dir)):
    data_dir = os.path.normpath(os.path.join(cur_dir, "data"))
if not (os.path.exists(data_dir) and os.path.isdir(data_dir)):
    data_dir = os.path.join(cur_dir, "_internal/data")
if not (os.path.exists(data_dir) and os.path.isdir(data_dir)):
    data_dir = "data"


def create_box_sorter():
    def sorter(image_file, detections: list[tuple[int, int, int, int]]):
        # what we need to do is to sort the detections by their y coordinate, and then their x coordinate
        # We find all the detections that have y coordinates that overlap by more than 50% of their height
        # We then sort those detections by their x coordinate
        # We then repeat this process until all detections are sorted
    
        # first we sort the detections by their y coordinate
        detections.sort(key=lambda x: x[1])

        detection_lists :list[list[tuple]] = []
        
        for detection in detections:
            added = False
            
            # we will iterate through the list of lists
            for detection_list in detection_lists:
                # we will check if the detection overlaps with any of the detections in the list
                for detection_in_list in detection_list:
                    # we will check if the detection overlaps with the detection in the list by more than 50 percent
                    if (detection[1] >= detection_in_list[1] and detection[1] <= detection_in_list[3]) or (detection[3] >= detection_in_list[1] and detection[3] <= detection_in_list[3]):
                        overlap_start = max(detection[1], detection_in_list[1])
                        overlap_end = min(detection[3], detection_in_list[3])
                        overlap_height = overlap_end - overlap_start
                        detection_height = detection[3] - detection[1]
                        detection_in_list_height = detection_in_list[3] - detection_in_list[1]
                        if overlap_height / detection_height >= 0.5 or overlap_height / detection_in_list_height >= 0.5:
                            detection_list.append(detection)
                            added = True
                            break
                if added:
                    break
            
            if not added:
                # we have not added the detection to a list
                # we will create a new list and add the detection to it
                detection_lists.append([detection])
                
        #sort by the x2 coordinate in reverse order
        for detection_list in detection_lists:
            detection_list.sort(key=lambda x: x[2], reverse=True)
            
        # cat them all together
        sorted_detections = []
        for detection_list in detection_lists:
            sorted_detections += detection_list
        
        return sorted_detections
    return sorter


def create_darknet_detector(detection_sorter):
    from .darknet import load_darknet_detector
    MODEL_CFG = os.path.join(data_dir, "model.cfg")
    MODEL_WEIGHTS = os.path.join(data_dir, "model.weights")

    print("Darknet: Using model.cfg: {}".format(MODEL_CFG))
    print("Darknet: Using model.weights: {}".format(MODEL_WEIGHTS))

    Detector = load_darknet_detector()
    detector = Detector(
        MODEL_CFG,
        MODEL_WEIGHTS,
        0)

    def detect(image_file):
        result = detector.detect(image_file)
        return [(x1 - 10, y1 - 10, x2 + 10, y2 + 10) for x1, y1, x2, y2 in detection_sorter(image_file, result)]

    return detect


class ogkalu_bbox:
    x: int
    y: int
    x2: int
    y2: int
    confidence: float
    label: int
    
    def __init__(self, x: int, y: int, x2: int, y2: int, confidence: float, label: int):
        self.x = int(x)
        self.y = int(y)
        self.x2 = int(x2)
        self.y2 = int(y2)
        self.confidence = float(confidence)
        self.label = int(label)


def create_ogkalu_detector(detection_sorter):
    import numpy as np
    import onnxruntime as ort
    from onnxruntime import InferenceSession
    from huggingface_hub import hf_hub_download, snapshot_download
    from PIL import Image
    
    model_name = "ogkalu/comic-text-and-bubble-detector"
    model_dir: str = snapshot_download(
        repo_id=model_name
    )
    model_filename = "detector.onnx"
    model_path = os.path.join(model_dir, model_filename)
    session: InferenceSession = InferenceSession(model_path)
        
    def read_image(path):
        """Read an image file and return as RGB numpy array."""
        im = Image.open(path)
        if im.mode != "RGB":
            im = im.convert("RGB")
        return im

    def get_image_array(path):
        pil_image = read_image(path)
        image_arr = np.array(pil_image)
        
        # pil_image = Image.fromarray(image_arr)  # image is already in RGB format
        im_resized = pil_image.resize((640, 640))
        arr = np.asarray(im_resized, dtype=np.float32) / 255.0  # (H,W,3)
        arr = np.transpose(arr, (2, 0, 1))  # (3,H,W)
        im_data = arr[np.newaxis, ...]  # (1,3,H,W)

        w, h = pil_image.size
        orig_size = np.array([[w, h]], dtype=np.int64)
        return im_data, orig_size, image_arr

    def process_detection(result: list[ogkalu_bbox]):
        return [(box.x, box.y, box.x2, box.y2) for box in result]

    def detect(image_file):
        resized_im_data, orig_size, image_arr = get_image_array(image_file)
                
        opt = {}
        opt["orig_target_sizes"] = [orig_size]
        outputs = session.run(None, {
        "images": resized_im_data,
        "orig_target_sizes": orig_size
        })
        names = {
            0: "bubble",
            1: "text_bubble",
            2: "text_free"
        }
        
        labels, boxes, scores = outputs[:3]

        if isinstance(labels, np.ndarray) and labels.ndim == 2 and labels.shape[0] == 1:
            labels = labels[0]
        if isinstance(scores, np.ndarray) and scores.ndim == 2 and scores.shape[0] == 1:
            scores = scores[0]
        if isinstance(boxes, np.ndarray) and boxes.ndim == 3 and boxes.shape[0] == 1:
            boxes = boxes[0]


        bubble_boxes = []
        text_boxes = []
        for i, box in enumerate(boxes):
            confidence = scores[i]
            if confidence < 0.3:
                continue
            label = labels[i]
            new_box = ogkalu_bbox(box[0], box[1], box[2], box[3], confidence, label)
            if label == 0:
                bubble_boxes.append(new_box)
            elif label in [1, 2]:
                text_boxes.append(new_box)
                
        result = process_detection(text_boxes)
        return [(x1, y1, x2, y2) for x1, y1, x2, y2 in detection_sorter(image_file, result)]

    return detect

def create_manga_ocr():
    from manga_ocr import MangaOcr
    mocr = MangaOcr(force_cpu=True)

    def ocr(image_file):
        with PIL.Image.open(image_file) as image:
            return mocr(image).replace("．．．", "…")

    return ocr


def create_tesseract_ocr():
    import pytesseract

    def ocr(image_file):
        with PIL.Image.open(image_file) as image:
            if image.width < image.height:
                lang = 'jpn+jpn_vert'
                config = '--psm 5'
            else:
                lang = 'jpn+jpn_vert'
                config = '--psm 6'
            text = pytesseract.pytesseract.image_to_string(image, lang='jpn+jpn_vert', config=config)
            return text

    return ocr


def create_ryou_ocr():
    import base64
    import requests

    def ocr(image_file):
        imageBase64 = base64.b64encode(image_file.read()).decode('ascii')
        r = requests.post(url='http://192.168.1.122:9846/api/ocr', json={ 'type': 'japanese', 'image': imageBase64 })
        text = "\n".join(r.json()['Lines'])
        return text

    return ocr


def create_combined_detector_ocr(ocr, detector):
    def combined(image_file):
        detections = detector(image_file)
        image_file.seek(0, io.SEEK_SET)
        with PIL.Image.open(image_file) as image:
            for detection in detections:
                region = image.crop(detection)
                with io.BytesIO() as region_file:
                    region.save(region_file, 'PNG')
                    region_file.seek(0, io.SEEK_SET)
                    text = ocr(region_file)
                    yield {"text": text, "rect": detection}

    return lambda image_file: list(combined(image_file))


def create_engines(
        ocr_mode: str,
        detector_mode: str,
        combined_mode: Union[str, None],
        detection_sorter_mode: str):

    if detection_sorter_mode == 'y_coordinate':
        sorter = create_box_sorter()

    if ocr_mode == 'manga-ocr':
        ocr = create_manga_ocr()
    elif ocr_mode == 'tesseract':
        ocr = create_tesseract_ocr()
    elif ocr_mode == 'ryou':
        ocr = create_ryou_ocr()
    else:
        ocr = None

    detector = None
    if detector_mode == 'darknet':
        try:
            detector = create_darknet_detector(sorter)
        except Exception as e:
            print(f"Error creating darknet detector: {e}")
            print(f"Falling back to ogkalu detector")
            detector = None
    if detector is None:
        detector = create_ogkalu_detector(sorter)

    if combined_mode is None:
        combined_detector_ocr = create_combined_detector_ocr(ocr, detector)

    return ocr, detector, combined_detector_ocr


def create_app(ocr, detector, combined_detector_ocr, config=None):
    app = Flask(__name__)
    app.config.update(DEBUG=True)
    app.config.update(config or {})

    CORS(app)

    @app.route("/ready", methods=['GET'])
    def is_ready():
        return jsonify({"ready": True})

    @app.route("/ocr", methods=['POST'])
    def issue_ocr():
        image_file = request.files['input_image']
        with io.BytesIO(image_file.stream.read()) as buffered:
            return jsonify(str(ocr(buffered)))

    @app.route("/detect", methods=['POST'])
    def issue_textbox_detection():
        image_file = request.files['input_image']
        with io.BytesIO(image_file.stream.read()) as buffered:
            return jsonify(detector(buffered))

    @app.route("/detect_ocr", methods=['POST'])
    def issue_textbox_detection_with_ocr():
        image_file = request.files['input_image']
        with io.BytesIO(image_file.stream.read()) as buffered:
            return jsonify(combined_detector_ocr(buffered))

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", action="store", default="127.0.0.1")
    parser.add_argument("--port", action="store", default="8000")
    parser.add_argument("--detection-mode", action="store", default="ogkalu")
    parser.add_argument("--ocr-mode", action="store", default="manga-ocr")
    parser.add_argument("--combined-detection-ocr-mode", action="store", default=None)
    parser.add_argument("--detection-ordering-mode", action="store", default='y_coordinate')

    args = parser.parse_args()
    ocr, detector, combined_detector_ocr = create_engines(
        args.ocr_mode,
        args.detection_mode,
        args.combined_detection_ocr_mode,
        args.detection_ordering_mode)
    created_app = create_app(ocr, detector, combined_detector_ocr)
    created_app.run(host=args.host, port=args.port, use_reloader=False)


if __name__ == "__main__":
    main()
