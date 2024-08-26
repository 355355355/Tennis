from ultralytics import YOLO
import cv2
import torch
import pickle
import pandas as pd
import pprint


class BallTracker:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path).to(self.device)

    # 对 ball_positions 列表进行插值与 backfill 操作，填补缺失值
    def interpolate_ball_positions(self, ball_positions):
        # 从每个元素中获取键为 1 的值，如果该键不存在，则返回一个空列表
        # 这一步操作后，ball_positions 变成了一个列表，每个元素是一个列表，包含了球的位置信息
        ball_positions = [x.get(1, []) for x in ball_positions]

        # df 是 DataFrame 的缩写
        df_ball_positions = pd.DataFrame(
            ball_positions, columns=["x1", "y1", "x2", "y2"]
        )
        # # 设置显示选项以完整打印 DataFrame
        # pd.set_option("display.max_rows", None)
        # pd.set_option("display.max_columns", None)
        # pd.set_option("display.max_colwidth", None)
        # pd.set_option("display.width", None)

        # pprint.pprint(df_ball_positions)
        # exit()
        # interpolate the missing values
        # 对 df_ball_positions 数据框进行插值操作，填补缺失值。插值方法默认使用线性插值。
        # 当有连续的缺失值时，插值会根据连续缺失值两边的非缺失值进行等差插值。
        df_ball_positions = df_ball_positions.interpolate()

        # 对 df_ball_positions 数据框进行向后填充操作，即用后一个非缺失值填补前面的缺失值。
        # 当有连续的缺失值时，向后填充会一直找到下一个非缺失值，然后用该值填充之前连续的缺失值。
        # 这一步是为了填补第一行的缺失值，因为插值操作无法填补第一行的缺失值。
        df_ball_positions = df_ball_positions.bfill()

        # 将 df_ball_positions 数据框转换为 NumPy 数组，然后转换为列表，最后将每个列表元素包装成一个字典，字典的键为 1，值为列表元素。
        ball_positions = [{1: x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    # 返回击球帧的 id 列表
    def get_ball_shot_positions(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        df = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])

        df["ball_hit"] = 0
        df["mid_y"] = (df["y1"] + df["y2"]) / 2

        # 以右对齐方式计算窗口大小为 5 的滚动均值
        # 通过计算 mid_y 列的滚动均值，可以减少数据中的噪声和短期波动，使得数据更加平滑和稳定。这有助于更准确地检测球的位置变化。
        df["mid_y_rolling_mean"] = (
            df["mid_y"].rolling(window=5, min_periods=1, center=False).mean()
        )

        df["delta_y"] = df["mid_y_rolling_mean"].diff()
        # # 设置显示选项以完整打印 DataFrame
        # pd.set_option("display.max_rows", None)
        # pd.set_option("display.max_columns", None)
        # pd.set_option("display.max_colwidth", None)
        # pd.set_option("display.width", None)
        # pprint.pprint(df)
        # exit()
        # minimum_change_frames_for_hit 表示在检测到球的位置变化时，至少需要有多少帧的变化才能确认这是一次击球。这有助于过滤掉噪声和短暂的变化，确保只有持续的、显著的变化才会被认为是一次击球。
        minimum_change_frames_for_hit = 25
        for i in range(1, len(df) - int(minimum_change_frames_for_hit * 1.2)):
            negative_position_change = (
                df["delta_y"].iloc[i] > 0 and df["delta_y"].iloc[i + 1] < 0
            )
            positive_position_change = (
                df["delta_y"].iloc[i] < 0 and df["delta_y"].iloc[i + 1] > 0
            )

            if negative_position_change or positive_position_change:
                change_count = 0
                for change_frame in range(
                    i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1
                ):
                    negative_position_change_following_frame = (
                        df["delta_y"].iloc[i] > 0
                        and df["delta_y"].iloc[change_frame] < 0
                    )
                    positive_position_change_following_frame = (
                        df["delta_y"].iloc[i] < 0
                        and df["delta_y"].iloc[change_frame] > 0
                    )

                    if (
                        negative_position_change
                        and negative_position_change_following_frame
                    ):
                        change_count += 1
                    elif (
                        positive_position_change
                        and positive_position_change_following_frame
                    ):
                        change_count += 1

                if change_count > minimum_change_frames_for_hit - 1:
                    df.loc[i, "ball_hit"] = 1

        frame_nums_with_ball_hit = df[df["ball_hit"] == 1].index.tolist()

        return frame_nums_with_ball_hit
 
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:  # "rb" 表示以二进制读模式打开文件
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            ball_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0]

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result

        return ball_dict

    def draw_bboxes(self, video_frames, ball_detections):
        output_video_frames = []
        for frame, ball_dict in zip(
            video_frames, ball_detections
        ):  # zip 能同时迭代两个序列
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(
                    frame,
                    f"Ball ID: {track_id}",
                    (int(bbox[0]), int(bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                )
                cv2.rectangle(
                    frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2
                )
            output_video_frames.append(frame)

        return output_video_frames
