from ultralytics import YOLO
import cv2
import torch
import pickle
import sys
import pprint

# 将上一级目录添加到 Python 的模块搜索路径中，从而能够导入上一级目录中的模块或包。
sys.path.append("..")
from utils import get_center_of_bbox, measure_distance


class PlayerTracker:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path).to(self.device)

    # 对 player_detections 进行只选取比赛的两个 player 的操作
    def choose_and_filter_players(self, court_keypoints, player_detections):
        first_frame_player_dict = player_detections[0]  # 选取第一帧的检测结果
        chosen_player = self.choose_players(court_keypoints, first_frame_player_dict)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {
                track_id: bbox  # 字典推导式的结果部分，表示将 track_id 作为键，bbox 作为值
                for track_id, bbox in player_dict.items()
                # 只有当 track_id 在 chosen_player 列表中时，才会将该 track_id 和对应的 bbox 添加到 filtered_player_dict 中。
                if track_id in chosen_player
            }
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    # 返回一个 list，包含了两个 player 的 id（根据与 14 个场线关键点距离的最小值选取）
    def choose_players(self, court_keypoints, player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)
            min_distance = float("inf")
            # 步长为 2 是因为每个关键点有两个坐标值
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i + 1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))

        # key 指定排序的依据，这里指定按照元组的第二个元素排序；sort 默认是升序排序
        distances.sort(key=lambda x: x[1])
        # choose the first 2 trackers
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players

    # 返回一个 list，每个元素是一个 dict，包含了每一帧检测到的球员的位置信息：键是 id，值是左上角和右下角坐标组成的 list
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:  # "rb" 表示以二进制读模式打开文件
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(player_detections, f)

        return player_detections

    # 返回一个 dict，键是检测到的 player 的 id，值是 bbox 的左上角和右下角坐标组成的 list
    def detect_frame(self, frame):
        # persist 参数用于控制在跟踪过程中是否保留中间结果：
        # 如果 persist=True，则跟踪模型会保留每一帧的跟踪结果，这样可以在后续的处理或分析中使用这些结果；
        # 如果 persist=False，则中间结果不会被保留，可能会节省内存，但无法进行后续的详细分析。
        result = self.model.track(frame, persist=True)[0]

        # result.names 是一个 dict，键是类别 id，值是类别名称
        id_name_dict = result.names

        player_dict = {}
        # box.id
        # id 表示跟踪对象的唯一标识符。每个被跟踪的对象都会分配一个唯一的 id，以便在视频帧之间保持对象的一致性。
        # 例如，如果在视频的第一个帧中检测到一个人并分配了 id 为 1，那么在后续帧中，即使这个人的位置发生了变化，只要他仍然被检测到，他的 id 仍然会是 1。
        # box.cls
        # cls 表示检测到的对象的类别，例如 cls: tensor([0.]) 表示检测到的对象类别为 0。
        for box in result.boxes:
            # tolist 是因为 box.id 是一个 tensor，tolist() 可以将 tensor 转换为 list
            # 例 id: tensor([1.])
            track_id = int(box.id.tolist()[0])
            # 例 xyxy: tensor([[575.0024, 560.8030, 633.7485, 688.5343]])，tolist 后变成 [[575.0024, 560.8030, 633.7485, 688.5343]]
            result = box.xyxy.tolist()[0]
            # 例 cls: tensor([0.])
            cls_id = box.cls.tolist()[0]
            cls_name = id_name_dict[cls_id]
            if cls_name == "person":
                player_dict[track_id] = result

        return player_dict

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(
            video_frames, player_detections
        ):  # zip 能同时迭代两个序列
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(
                    frame,
                    f"Player ID: {track_id}",
                    (int(bbox[0]), int(bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )
                cv2.rectangle(
                    frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2
                )
            output_video_frames.append(frame)

        return output_video_frames
