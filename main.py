import cv2
from ultralytics import YOLO
from datetime import datetime
import os
import sys
import argparse
import contextlib


CURRENT_DIR_PATH = os.getcwd()
OUTPUT_DIR_NAME = "video_recordings"


@contextlib.contextmanager
def suppress_stderr():
    # Redirects low-level stderr from OpenCV's C++ backend
    with open(os.devnull, "w") as devnull:
        old_stderr_fd = os.dup(2)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stderr_fd, 2)
            os.close(old_stderr_fd)


def list_webcam_devices(max_index=10):
    available = []
    with suppress_stderr():
        for i in range(max_index):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(i)
                cap.release()
    print("🎥 Available webcam indexes:", available)


class Cholo_Bump_Dekhi:
    def __init__(self, model_path, webcam_device=0, record=False):
        print("USING OpenCV Version:", cv2.__version__)
        print("USING YOLO Version:", YOLO._version)

        self.webcam_device = webcam_device
        self.cap = cv2.VideoCapture(self.webcam_device)
        self.record = record
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        self.model_path = model_path
        self.model = self.get_model()
        self.file_name = self.get_file_name()

    @property
    def fps(self):
        fps_ = int(
            self.cap.get(cv2.CAP_PROP_FPS)
        )  # Get FPS from webcam (default is 30 if 0)
        if fps_ == 0:
            fps_ = 30  # Set a default FPS if camera doesn't provide it
        return fps_

    def get_model(self):
        assert os.path.exists(self.model_path), "Model Path doesn't exist!"
        return YOLO(self.model_path)

    def get_file_name(self):
        now = datetime.now()
        now_str = now.strftime("%m_%d_%Y-%H_%M_%S")
        name_base = "road_damage_detection"
        outfile_name = f"{name_base}-{now_str}.mp4"
        output_path = os.path.join(CURRENT_DIR_PATH, OUTPUT_DIR_NAME)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        output_file_path = os.path.join(output_path, outfile_name)
        return output_file_path

    def start_video(self):
        if self.record:
            # Mp4 Video codec and video writer record korar jonno
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                self.file_name, fourcc, self.fps, (self.frame_width, self.frame_height)
            )
        print("Video Started...Press `q` to stop and exit")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break  # Stop if no frame is self.captured

            # The magic!
            results = self.model(frame, verbose=False)  # Perform inference

            # bounding box logics
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
                    conf = float(box.conf[0])  # Confidence score
                    class_ = int(box.cls[0])  # Class index
                    label = f"{self.model.names[class_]}: {conf:.2f}"

                    # Draw rectangle & label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

            # Write the frame
            if self.record:
                video_writer.write(frame)

            # Video Display
            cv2.imshow("Road Damage Detection", frame)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                if self.record:
                    print(f"Video Saved at: {self.file_name}")
                break

        # Clear resources
        self.cap.release()
        video_writer.release() if self.record else ...
        cv2.destroyAllWindows()


def command_line_interface():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: List
    parser_list = subparsers.add_parser("list", help="List available index of webcam devices")

    # Subcommand: Run
    parser_run = subparsers.add_parser("run", help="Run the model on webcam")
    parser_run.add_argument("model_path", type=str, help="Path to the model")
    parser_run.add_argument("-wc", "--webcam", type=int, help="Webcam device index")
    parser_run.add_argument(
        "-r",
        "--record",
        action="store_true",
        help="If passed, the video will be recorded and saved to ./recorded_videos/",
    )

    args = parser.parse_args()

    if args.command == "list":
        list_webcam_devices()

    elif args.command == "run":
        app = Cholo_Bump_Dekhi(args.model_path, webcam_device=args.webcam, record=args.record)
        app.start_video()


if __name__ == "__main__":
    command_line_interface()
