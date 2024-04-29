import os
import time
import cv2
import math

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort


class ObjectDetection:

    def __init__(self, capture, fps=10, conf_threshold=0.6):
        self.capture = capture
        self.fps = fps
        self.conf_threshold = conf_threshold
        self.model = self.load_model()

    @staticmethod
    def load_model():
        model = YOLO('best1.pt')
        model.fuse()

        return model

    def predict(self, img):
        results = self.model(img, stream=True)

        return results

    def plot_boxes(self, results, img):
        detections = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                cls = int(box.cls[0])
                currentclass = self.CLASS_NAMES_DICT[cls]

                conf = math.ceil(box.conf[0] * 100) / 100

                if conf > self.conf_threshold:
                    detections.append((([x1, y1, w, h]), conf, currentclass))

        return detections, img

    @staticmethod
    def track_detect(detections, img, tracker):
        tracks = tracker.update_tracks(detections, frame=img)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()

            x1, y1, x2, y2 = map(int, ltrb)

            cv2.putText(img, f'ID: {track_id}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

    def __call__(self):
        cap = cv2.VideoCapture(self.capture)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        assert cap.isOpened()
        cfg = get_config()
        cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
        tracker = DeepSort(cfg.DEEPSORT.REID_CKPT,
                           max_dist=cfg.DEEPSORT.MAX_DIST,
                           min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                           nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                           max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                           max_age=cfg.DEEPSORT.MAX_AGE,
                           n_init=cfg.DEEPSORT.N_INIT,
                           nn_budget=cfg.DEEPSORT.NN_BUDGET,
                           use_cuda=True)

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        fourcc = cv2.VideoWriter.fourcc(*'XVID')
        if not os.path.exists('video_run'):
            os.makedirs('video_run')
        out = cv2.VideoWriter('video_run/output.mp4', fourcc, self.fps, (int(width), int(height)))

        prev_frame_time = 0

        while True:
            ret, img = cap.read()
            if not ret:
                print("已读取完毕")
                break

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)
            fps = str(fps)
            cv2.putText(img, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

            results = self.predict(img)
            detections, frames = self.plot_boxes(results, img)
            detect_frame = self.track_detect(detections, frames, tracker)

            cv2.imshow('Image', detect_frame)
            cv2.namedWindow('Image', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            out.write(detect_frame)

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()


detector = ObjectDetection(capture='drone.mp4')  # input your video link
detector()
