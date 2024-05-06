import cv2
from ultralytics import YOLO
import math
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import os


class ObjectDetection:

    def __init__(self, capture, fps=30, conf_threshold=0.3):
        self.capture = capture
        self.fps = fps
        self.conf_threshold = conf_threshold
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.names

    def load_model(self):
        model = YOLO('bestv9.pt')
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
                currentClass = self.CLASS_NAMES_DICT[cls]

                conf = math.ceil(box.conf[0] * 100) / 100

                if conf > self.conf_threshold:
                    detections.append((([x1, y1, w, h]), conf, currentClass))

        return detections, img

    def track_detect(self, detections, img, tracker):
        tracks = tracker.update_tracks(detections, frame=img)

        for track in tracks:
            if not track.is_confirmed():
                continue

            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cv2.putText(img, f'ID: {track.track_id}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

        return img

    def __call__(self):
        cap = cv2.VideoCapture(self.capture)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        assert cap.isOpened()
        tracker = DeepSort(max_age=5,
                           n_init=2,
                           nms_max_overlap=1.0,
                           max_cosine_distance=0.3,
                           nn_budget=None,
                           override_track_class=None,
                           embedder="mobilenet",
                           half=True,
                           bgr=True,
                           embedder_gpu=True,
                           embedder_model_name=None,
                           embedder_wts=None,
                           polygon=False,
                           today=None)

        # Get the width and height of the frame
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
        if not os.path.exists('video_run'):
            os.makedirs('video_run')
        out = cv2.VideoWriter('video_run/output.mp4', fourcc, self.fps, (int(width), int(height)))

        prev_frame_time = 0
        new_frame_time = 0

        while True:
            ret, img = cap.read()
            if not ret:
                print("已读取完毕")
                break

            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
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




