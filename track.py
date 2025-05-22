import cv2
import numpy as np
import math
import time
from ultralytics import YOLO
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.tools import generate_detections as gdet
from PIL import Image
import numpy as np
import requests
import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation


def download_video_from_drive(file_id, output_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_path, quiet=False)

# Chuyển đổi tọa độ bounding box
def xyxy_to_xywh(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    return x_c, y_c, bbox_w, bbox_h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for box in bbox_xyxy:
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_bboxs.append([top, left, w, h])
    return tlwh_bboxs



# Phương pháp 1: Hiệu chuẩn dựa trên vùng
class RegionBasedCalibration:
    def __init__(self, height, num_regions=10):
        self.height = height
        self.num_regions = num_regions
        self.region_height = height / num_regions
        # Hệ số PPM giảm dần từ dưới lên trên (từ gần đến xa)
        self.ppm_values = np.linspace(8, 2, num_regions)  # Từ 8 (gần) đến 2 (xa)
        
    def get_region(self, y):
        # Lấy chỉ số vùng dựa trên tọa độ y
        # Đảm bảo y nằm trong phạm vi hợp lệ
        y = max(0, min(y, self.height - 1))
        region_idx = int(y / self.region_height)
        # Đảm bảo chỉ số không vượt quá số lượng vùng
        region_idx = min(region_idx, self.num_regions - 1)
        return region_idx
        
    def get_ppm(self, y):
        region_idx = self.get_region(y)
        return self.ppm_values[region_idx]
    
    def estimate_speed(self, location1, location2, frames_between=1, fps=30):
        # Lấy PPM cho mỗi vị trí
        ppm1 = self.get_ppm(location1[1])
        ppm2 = self.get_ppm(location2[1])
        
        # Tính PPM trung bình
        ppm_avg = (ppm1 + ppm2) / 2
        
        # Tính khoảng cách pixel
        d_pixel = math.hypot(location2[0] - location1[0], location2[1] - location1[1])
        
        # Chuyển đổi sang mét
        d_meters = d_pixel / ppm_avg
        
        # Tính thời gian (giây)
        time_seconds = frames_between / fps
        
        # Tính tốc độ (km/h)
        speed = (d_meters / time_seconds) * 3.6
        
        return speed, ppm_avg

# Phương pháp 2: Sử dụng đường thẳng tham chiếu
class ReferenceLineSpeedEstimator:
    def __init__(self, reference_lines, real_distances, fps=30):
        """
        Khởi tạo với các đường tham chiếu
        reference_lines: Danh sách các tọa độ y của đường tham chiếu
        real_distances: Khoảng cách thực tế giữa các đường tham chiếu (mét)
        """
        self.reference_lines = reference_lines
        self.real_distances = real_distances
        self.fps = fps
        self.crossing_times = {}  # {track_id: {line_idx: frame_number}}
        self.line_speeds = {}  # {track_id: {line_pair: speed}}
        
    def update(self, track_id, position, previous_position, current_frame):
        """
        Cập nhật và tính tốc độ khi xe đi qua các đường tham chiếu
        """
        if track_id not in self.crossing_times:
            self.crossing_times[track_id] = {}
            self.line_speeds[track_id] = {}
        
        # Kiểm tra xem xe có đi qua đường tham chiếu nào không
        for i, line_y in enumerate(self.reference_lines):
            # Kiểm tra nếu xe đi qua đường từ trên xuống hoặc từ dưới lên
            crossed_down = previous_position[1] < line_y and position[1] >= line_y
            crossed_up = previous_position[1] > line_y and position[1] <= line_y
            
            if crossed_down or crossed_up:
                # Ghi nhận thời điểm đi qua đường tham chiếu
                self.crossing_times[track_id][i] = current_frame
                
                # Tính tốc độ nếu đã đi qua 2 đường tham chiếu
                for j in self.crossing_times[track_id]:
                    if j != i and abs(j - i) == 1:  # Đường liền kề
                        # Lấy chỉ số nhỏ hơn
                        min_idx = min(i, j)
                        frames_diff = abs(self.crossing_times[track_id][i] - self.crossing_times[track_id][j])
                        time_diff = frames_diff / self.fps
                        
                        if time_diff > 0:
                            # Lấy khoảng cách thực tế giữa hai đường
                            distance = self.real_distances[min_idx]
                            # Tính tốc độ (km/h)
                            speed = (distance / time_diff) * 3.6
                            # Lưu tốc độ
                            line_pair = (min(i, j), max(i, j))
                            self.line_speeds[track_id][line_pair] = speed
        
        return self.get_average_speed(track_id)
    
    def get_average_speed(self, track_id):
        """
        Lấy tốc độ trung bình từ tất cả các cặp đường tham chiếu đã đi qua
        """
        if track_id in self.line_speeds and self.line_speeds[track_id]:
            speeds = list(self.line_speeds[track_id].values())
            # Lấy 3 giá trị tốc độ gần nhất nếu có
            recent_speeds = speeds[-3:] if len(speeds) > 3 else speeds
            return np.mean(recent_speeds)
        return None

# Kết hợp hai phương pháp
class SpeedEstimator:
    def __init__(self, height, reference_lines, real_distances, fps=30):
        self.region_calibration = RegionBasedCalibration(height)
        self.reference_line_estimator = ReferenceLineSpeedEstimator(reference_lines, real_distances, fps)
        self.track_speeds = {}  # {track_id: [speed1, speed2, ...]}
        self.max_history = 5  # Số lượng tốc độ lưu trữ tối đa
        
    def update(self, track_id, position, previous_position, current_frame, previous_frame, fps):
        # Phương pháp 1: Dựa trên vùng
        frames_between = current_frame - previous_frame
        if frames_between <= 0:
            frames_between = 1
            
        region_speed, ppm = self.region_calibration.estimate_speed(
            previous_position, position, frames_between, fps
        )
        
        # Phương pháp 2: Dựa trên đường tham chiếu
        line_speed = self.reference_line_estimator.update(
            track_id, position, previous_position, current_frame
        )
        
        # Khởi tạo history nếu chưa có
        if track_id not in self.track_speeds:
            self.track_speeds[track_id] = []
            
        # Ưu tiên phương pháp đường tham chiếu nếu có
        if line_speed is not None:
            final_speed = line_speed
        else:
            final_speed = region_speed
            
        # Thêm vào lịch sử
        self.track_speeds[track_id].append(final_speed)
        
        # Giới hạn kích thước lịch sử
        if len(self.track_speeds[track_id]) > self.max_history:
            self.track_speeds[track_id].pop(0)
            
        # Lấy tốc độ trung bình từ lịch sử
        avg_speed = np.mean(self.track_speeds[track_id])
        
        return avg_speed, ppm

# Hàm xử lý video
def process_video(video_path, output_path, reference_lines=[200, 300, 400, 500], real_distances=[15, 15, 15]):
    # Khởi tạo YOLOv11
    model = YOLO('models/yolov11n.pt')

    # Khởi tạo DeepSORT
    max_cosine_distance = 0.2
    nn_budget = None
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    model_filename = "deep_sort/deep_checkpoint/mars-small128.pb"
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    # Mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video {video_path}")
        return

    # Lấy thông tin video
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Chiều cao khung hình: {height} pixel")  # Log chiều cao khung hình
    
    # Khởi tạo video đầu ra
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Khởi tạo bộ ước tính tốc độ
    speed_estimator = SpeedEstimator(height, reference_lines, real_distances, fps)

    # Khởi tạo biến
    vehicle_count = {'car': 0, 'bus': 0, 'truck': 0}
    counted_ids = {'car': set(), 'bus': set(), 'truck': set()}
    track_history = {}
    in_count = 0
    out_count = 0
    class_names = {2: 'car', 5: 'bus', 7: 'truck'}
    colors = {'car': (0, 0, 255), 'bus': (0, 255, 255), 'truck': (255, 0, 0)}  # Car: xanh dương, Bus: vàng, Truck: đỏ

    # Bắt đầu đo thời gian xử lý
    start_time = time.time()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Đã xử lý {frame_count} frames trong {elapsed:.2f} giây ({frame_count/elapsed:.2f} FPS)")

        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

        # Phát hiện đối tượng
        results = model(frame)
        boxes = []
        scores = []
        classes = []
        for r in results[0].boxes:
            cls = int(r.cls[0])
            conf = float(r.conf[0])
            if cls in [2, 5, 7] and conf > 0.5:
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                scores.append(conf)
                classes.append(cls)

        # Trích xuất đặc trưng
        features = encoder(frame, boxes)
        detections = [Detection(bbox, score, feature) for bbox, score, feature in zip(boxes, scores, features)]

        # Cập nhật tracker
        tracker.predict()
        tracker.update(detections)

        # Gán class (mặc định là car nếu không xác định)
        for track, cls in zip(tracker.tracks, classes):
            if track.is_confirmed() and track.time_since_update <= 1:
                track.det_class = cls if cls in class_names else 2  # Mặc định là car nếu không rõ

        # Vẽ các đường tham chiếu
        for line_y in reference_lines:
            cv2.line(frame, (0, line_y), (width, line_y), (255, 0, 0), 2)

        # Xử lý track
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()
            track_id = track.track_id
            cls = getattr(track, 'det_class', 2)  # Mặc định là car
            label = class_names.get(cls, 'car')

            # Đếm xe
            if track_id not in counted_ids[label]:
                vehicle_count[label] += 1
                counted_ids[label].add(track_id)

            # Tính tâm
            tlwh = xyxy_to_tlwh([bbox])[0]
            center_x = tlwh[0] + tlwh[2] / 2
            center_y = tlwh[1] + tlwh[3] / 2
            center = (center_x, center_y)

            # Tính tốc độ
            speed = 0
            ppm = 0
            if track_id in track_history:
                prev_center = track_history[track_id]['center']
                prev_frame = track_history[track_id]['frame']
                speed, ppm = speed_estimator.update(
                    track_id, center, prev_center, current_frame, prev_frame, fps
                )

                # Đếm vào/ra (sử dụng đường ở giữa để đếm)
                middle_line = reference_lines[len(reference_lines) // 2]
                prev_y = prev_center[1]
                curr_y = center_y
                if prev_y < middle_line and curr_y >= middle_line:
                    in_count += 1
                elif prev_y > middle_line and curr_y <= middle_line:
                    out_count += 1

            # Cập nhật lịch sử
            track_history[track_id] = {'center': center, 'frame': current_frame}

            # Vẽ khung viền
            color = colors[label]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

            # Hiển thị nhãn và tốc độ
            label_text = f"{label} {track_id}"
            speed_text = f"{speed:.1f} km/h"
            cv2.putText(frame, label_text, (int(bbox[0]), int(bbox[1]) - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, speed_text, (int(bbox[0]), int(bbox[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Debug info
            # cv2.putText(frame, f"PPM: {ppm:.2f}", (int(bbox[0]), int(bbox[1]) + 20),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Hiển thị số lượng
        cv2.putText(frame, f"Cars: {vehicle_count['car']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Buses: {vehicle_count['bus']}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Trucks: {vehicle_count['truck']}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"In: {in_count}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Out: {out_count}", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Ghi và hiển thị
        out.write(frame)
        #cv2.imshow('Vehicle Tracking', frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
         #   break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Số lượng xe: {vehicle_count}")
    print(f"Vào: {in_count}, Ra: {out_count}")
    print(f"Tổng thời gian xử lý: {time.time() - start_time:.2f} giây")
    print(f"FPS trung bình: {frame_count / (time.time() - start_time):.2f}")

def main(
    input_drive_file_id,
    output_filename="output.mp4",
    drive_output_folder_id=None
):
    # 1. Tải video từ Google Drive
    input_path = "input_video.mp4"
    download_video_from_drive(input_drive_file_id, input_path)

    # 2. Load model
    model = YOLO("yolov8n.pt")
    tracker = DeepSort()

    # 3. Process video
    process_video(model, tracker, input_path, output_filename)

    # 4. Upload kết quả lên Google Drive
    upload_file_to_drive(output_filename, drive_output_folder_id)


if __name__ == "__main__":
    # Nhập ID của video trên Google Drive
    input_drive_file_id = "YOUR_INPUT_VIDEO_ID"
    
    # Nếu muốn lưu vào folder cụ thể, nhập ID folder Drive, không thì để None
    drive_output_folder_id = "YOUR_OUTPUT_FOLDER_ID"  # or None

    main(input_drive_file_id, "output.mp4", drive_output_folder_id)
