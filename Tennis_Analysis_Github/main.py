from utils import (
    read_video,
    save_video,
    measure_distance,
    draw_player_stats,
    convert_pixel_distance_to_meters,
)
import constants
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2
from copy import deepcopy
import pandas as pd
import pprint


def main():
    # 读入视频
    input_video_path = "input_videos/input_video1.mp4"
    video_capture = cv2.VideoCapture(input_video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    print(f"视频帧率: {fps}")
    video_frames = read_video(input_video_path)

    # 检测击球人与球
    player_tracker = PlayerTracker(model_path="yolov8x")
    ball_tracker = BallTracker(model_path="models/yolov8n.pt")
    # player_detections 类型为 list；每个元素是一个 dict，包含了每一帧检测到的球员的位置信息：键是 id，值是左上角和右下角坐标组成的 list
    player_detections = player_tracker.detect_frames(
        video_frames,
        read_from_stub=True,
        stub_path="tracker_stubs/player_detections.pkl",
    )
    # ball_detections 类型为 list；每个元素是一个 dict，包含了每一帧检测到的球的位置信息：键是 id，值是左上角和右下角坐标组成的 list
    ball_detections = ball_tracker.detect_frames(
        video_frames,
        read_from_stub=True,
        stub_path="tracker_stubs/yolov8n.pkl",
    )
    frame_count = sum(1 for detection in ball_detections if detection)
    print(f"总帧数：{len(ball_detections)} 检测到球的帧数: {frame_count}")

    # 插值法保证每一帧都有球的位置信息，即使有些帧检测不到球。
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    # 检测场线关键点
    court_model_path = "models/keypoints.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    # court_keypoints 类型为 numpy.ndarray，共 14 * 2 个坐标值
    court_keypoints = court_line_detector.predict(video_frames[0])

    # 留下 player_detections 中进行比赛的人（即球员）
    player_detections = player_tracker.choose_and_filter_players(
        court_keypoints, player_detections
    )

    mini_court = MiniCourt(video_frames[0])

    # 检测球的击球帧
    ball_shot_frames = ball_tracker.get_ball_shot_positions(ball_detections)

    # 将球员和球的位置信息转换为 mini_court 上的坐标
    player_mini_court_detections, ball_mini_court_detections = (
        mini_court.convert_bounding_boxes_to_mini_court_coordinates(
            player_detections, ball_detections, court_keypoints
        )
    )

    player_stats_data = [
        {
            "frame_num": 0,  # 当前帧的编号
            #
            "player_1_number_of_shots": 0,  # 球员1的击球次数
            "player_1_total_shot_speed": 0,  # 球员1的总击球速度（所有击球速度的累加）
            "player_1_last_shot_speed": 0,  # 球员1最近一次击球速度
            "player_1_total_player_speed": 0,  # 球员 1 的总移动速度（所有移动速度的累加）
            "player_1_last_player_speed": 0,  # 球员 1 的最近一次移动速度
            #
            "player_2_number_of_shots": 0,  # 同理球员 1
            "player_2_total_shot_speed": 0,
            "player_2_last_shot_speed": 0,
            "player_2_total_player_speed": 0,
            "player_2_last_player_speed": 0,
        }
    ]

    # 注意只在击球帧上计算了球员的速度与击球速度
    for ball_shot_ind in range(len(ball_shot_frames) - 1):
        # 计算击球时间
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind + 1]
        ball_shot_time_in_seconds = (end_frame - start_frame) / fps

        # 计算球飞过的距离
        distance_covered_by_ball_pixels = measure_distance(
            ball_mini_court_detections[start_frame][1],
            ball_mini_court_detections[end_frame][1],
        )
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(
            distance_covered_by_ball_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court(),
        )

        # 以 km/h 为单位计算球的速度
        speed_of_ball_shot = (
            distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6
        )

        # 计算击球球员的 id
        player_positions = player_mini_court_detections[start_frame]
        player_shot_ball = min(
            player_positions.keys(),
            key=lambda player_id: measure_distance(
                player_positions[player_id], ball_mini_court_detections[start_frame][1]
            ),
        )

        # 仿照计算球的速度，计算非击球球员的速度
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_pixels = measure_distance(
            player_mini_court_detections[start_frame][opponent_player_id],
            player_mini_court_detections[end_frame][opponent_player_id],
        )
        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(
            distance_covered_by_opponent_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court(),
        )
        speed_of_opponent = (
            distance_covered_by_opponent_meters / ball_shot_time_in_seconds * 3.6
        )

        # 这里用 [-1] 的原因是 player_stats_data.append(current_player_stats)
        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats["frame_num"] = start_frame
        current_player_stats[f"player_{player_shot_ball}_number_of_shots"] += 1
        current_player_stats[
            f"player_{player_shot_ball}_total_shot_speed"
        ] += speed_of_ball_shot
        current_player_stats[f"player_{player_shot_ball}_last_shot_speed"] = (
            speed_of_ball_shot
        )

        current_player_stats[
            f"player_{opponent_player_id}_total_player_speed"
        ] += speed_of_opponent
        current_player_stats[f"player_{opponent_player_id}_last_player_speed"] = (
            speed_of_opponent
        )

        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)

    frames_df = pd.DataFrame({"frame_num": list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(
        frames_df, player_stats_data_df, on="frame_num", how="left"
    )
    # 用前面的值填充 NaN
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df["player_1_average_shot_speed"] = (
        player_stats_data_df["player_1_total_shot_speed"]
        / player_stats_data_df["player_1_number_of_shots"]
    )
    player_stats_data_df["player_2_average_shot_speed"] = (
        player_stats_data_df["player_2_total_shot_speed"]
        / player_stats_data_df["player_2_number_of_shots"]
    )
    player_stats_data_df["player_1_average_player_speed"] = (
        player_stats_data_df["player_1_total_player_speed"]
        / player_stats_data_df["player_2_number_of_shots"]
    )
    player_stats_data_df["player_2_average_player_speed"] = (
        player_stats_data_df["player_2_total_player_speed"]
        / player_stats_data_df["player_2_number_of_shots"]
    )

    # 画出球员和球的锚框
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(video_frames, ball_detections)
    # 在 mini_court 画出场线关键点
    output_video_frames = court_line_detector.draw_keypoints_on_video(
        output_video_frames, court_keypoints
    )
    # 在 mini_court 画出球员位置
    output_video_frames = mini_court.draw_points_on_mini_court(
        output_video_frames, player_mini_court_detections
    )
    # 在 mini_court 画出球的位置
    output_video_frames = mini_court.draw_points_on_mini_court(
        output_video_frames, ball_mini_court_detections, color=(0, 255, 255)
    )
    # 在帧上画 mini_court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    # 在 mini_court 上以绿色画出球员
    output_video_frames = mini_court.draw_points_on_mini_court(
        output_video_frames, player_mini_court_detections
    )
    # 在 mini_court 上以黄色画出球
    output_video_frames = mini_court.draw_points_on_mini_court(
        output_video_frames, ball_mini_court_detections, color=(0, 255, 255)
    )
    # Draw player stats
    output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)

    ## Draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(
            frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

    save_video(
        output_video_frames,
        "output_videos/output_video1.avi",
    )


if __name__ == "__main__":
    main()
