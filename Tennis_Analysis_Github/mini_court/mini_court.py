import cv2
import sys
import numpy as np

sys.path.append("../")
import constants
from utils import (
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters,
    get_foot_position,
    get_closest_keypoint_index,
    get_height_of_bbox,
    measure_xy_distance,
    get_center_of_bbox,
    measure_distance,
)


class MiniCourt:
    def __init__(self, frame):
        self.drawing_rectangle_width = 150  # 画布的宽度
        self.drawing_rectangle_height = 270
        self.buffer = 40  # 画布的边缘距离 frame 的距离
        self.padding_court = 20  # 球场方框距离画布边缘的距离

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()

    def convert_meters_to_pixels(self, meters):
        return convert_meters_to_pixel_distance(
            meters,
            constants.DOUBLE_LINE_WIDTH,
            self.court_drawing_width,
        )

    # 设置 frame 中的球场关键点对应在 mini_court 上的点于 frame 上的位置
    def set_court_drawing_key_points(self):
        drawing_key_points = [0] * 28

        # point 0
        drawing_key_points[0], drawing_key_points[1] = int(self.court_start_x), int(
            self.court_start_y
        )
        # point 1
        drawing_key_points[2], drawing_key_points[3] = int(self.court_end_x), int(
            self.court_start_y
        )
        # point 2
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = self.court_start_y + self.convert_meters_to_pixels(
            constants.HALF_COURT_LINE_HEIGHT * 2
        )
        # point 3
        drawing_key_points[6] = int(self.court_end_x)
        drawing_key_points[7] = drawing_key_points[5]
        # point 4
        drawing_key_points[8] = drawing_key_points[0] + self.convert_meters_to_pixels(
            constants.DOUBLE_ALLY_DIFFERENCE
        )
        drawing_key_points[9] = drawing_key_points[1]
        # point 5
        drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_to_pixels(
            constants.DOUBLE_ALLY_DIFFERENCE
        )
        drawing_key_points[11] = drawing_key_points[5]
        # point 6
        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(
            constants.DOUBLE_ALLY_DIFFERENCE
        )
        drawing_key_points[13] = drawing_key_points[3]
        # point 7
        drawing_key_points[14] = drawing_key_points[6] - self.convert_meters_to_pixels(
            constants.DOUBLE_ALLY_DIFFERENCE
        )
        drawing_key_points[15] = drawing_key_points[7]
        # point 8
        drawing_key_points[16] = drawing_key_points[8]
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(
            constants.NO_MANS_LAND_HEIGHT
        )
        # point 9
        drawing_key_points[18] = drawing_key_points[16] + self.convert_meters_to_pixels(
            constants.SINGLE_LINE_WIDTH
        )
        drawing_key_points[19] = drawing_key_points[17]
        # point 10
        drawing_key_points[20] = drawing_key_points[10]
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(
            constants.NO_MANS_LAND_HEIGHT
        )
        # point 11
        drawing_key_points[22] = drawing_key_points[20] + self.convert_meters_to_pixels(
            constants.SINGLE_LINE_WIDTH
        )
        drawing_key_points[23] = drawing_key_points[21]
        # point 12
        drawing_key_points[24] = int(
            (drawing_key_points[16] + drawing_key_points[18]) / 2
        )
        drawing_key_points[25] = drawing_key_points[17]
        # point 13
        drawing_key_points[26] = int(
            (drawing_key_points[20] + drawing_key_points[22]) / 2
        )
        drawing_key_points[27] = drawing_key_points[21]

        self.drawing_key_points = drawing_key_points

    # 设置应由哪些点对来组成线
    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6, 7),
            (1, 3),
            (0, 1),
            (8, 9),
            (10, 11),
            (2, 3),
        ]

    # 设置 mini_court 的位置
    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x

    # 设置 canvas 在 frame 中的位置
    def set_canvas_background_box_position(self, frame):
        frame = frame.copy()  # 创建副本，避免对原始 frame 的修改

        # frame.shape = (height, width, channels) = (720, 1280, 3)
        # 原点在左上角，x 轴向右，y 轴向下
        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.buffer

    def draw_court(self, frame):
        for i in range(0, len(self.drawing_key_points), 2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i + 1])
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        # Draw lines
        for line in self.lines:
            start_point = (
                int(self.drawing_key_points[line[0] * 2]),
                int(self.drawing_key_points[line[0] * 2 + 1]),
            )
            end_point = (
                int(self.drawing_key_points[line[1] * 2]),
                int(self.drawing_key_points[line[1] * 2 + 1]),
            )
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        # Draw net
        net_start_point = (
            self.drawing_key_points[0],
            int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2),
        )
        net_end_point = (
            self.drawing_key_points[2],
            int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2),
        )
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)

        return frame

    def draw_background_rectangle(self, frame):
        # 创建一个与 frame 大小相同的全零数组，并将其数据类型设置为 np.uint8。
        # 主要目的是创建一个空白的图像（全黑图像），其大小和形状与 frame 相同。这个空白图像可以用于绘制形状、标注或其他图像处理操作，而不会影响原始的 frame 图像。
        shapes = np.zeros_like(frame, np.uint8)
        cv2.rectangle(
            shapes,
            (self.start_x, self.start_y),
            (self.end_x, self.end_y),
            (255, 255, 255),  # 白
            cv2.FILLED,
        )
        # 创建副本，这样可以在不修改原始图像的情况下进行操作
        out = frame.copy()
        alpha = 0.5
        # 将 shapes 数组转换为布尔类型数组 mask,非零值转换为 True，零值转换为 False
        mask = shapes.astype(bool)
        # 使用 cv2.addWeighted 将 frame 和 shapes 进行加权合成。alpha 和 1 - alpha 分别是 frame 和 shapes 的权重。
        # 0 是 gamma 值，gamma 值是一个标量，它会加到加权和上。
        # mask 用于指定图像中需要进行加权求和操作的区域： True 表示需要进行操作的像素位置，False 表示不需要进行操作的像素位置。
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
        return out

    def draw_mini_court(self, frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames

    def get_start_point_of_mini_court(self):
        return (self.court_start_x, self.court_start_y)

    def get_width_of_mini_court(self):
        return self.court_drawing_width

    def get_court_drawing_keypoints(self):
        return self.drawing_key_points

    # 返回球员在 mini_court 上的 frame 坐标
    def get_mini_court_coordinates(
        self,
        object_position,
        closest_key_point,
        closest_key_point_index,
        player_height_in_pixels,
        player_height_in_meters,
    ):
        distance_from_keypoint_x_pixels, distance_from_keypoint_y_pixels = (
            measure_xy_distance(object_position, closest_key_point)
        )

        # 把 frame 上的距离转换为实际的距离
        distance_from_keypoint_x_meters = convert_pixel_distance_to_meters(
            distance_from_keypoint_x_pixels,
            player_height_in_meters,
            player_height_in_pixels,
        )
        distance_from_keypoint_y_meters = convert_pixel_distance_to_meters(
            distance_from_keypoint_y_pixels,
            player_height_in_meters,
            player_height_in_pixels,
        )
        # 把实际的距离转换为 mini_court 上的距离
        mini_court_x_distance_pixels = self.convert_meters_to_pixels(
            distance_from_keypoint_x_meters
        )
        mini_court_y_distance_pixels = self.convert_meters_to_pixels(
            distance_from_keypoint_y_meters
        )
        closest_mini_court_keypoint = (
            self.drawing_key_points[closest_key_point_index * 2],
            self.drawing_key_points[closest_key_point_index * 2 + 1],
        )
        mini_court_player_position = (
            closest_mini_court_keypoint[0] + mini_court_x_distance_pixels,
            closest_mini_court_keypoint[1] + mini_court_y_distance_pixels,
        )

        return mini_court_player_position

    # 返回球员和球在 mini_court 上的字典列表
    def convert_bounding_boxes_to_mini_court_coordinates(
        self, player_boxes, ball_boxes, original_court_key_points
    ):
        player_heights = {
            1: constants.PLAYER_1_HEIGHT_METERS,
            2: constants.PLAYER_2_HEIGHT_METERS,
        }

        output_player_boxes = []
        output_ball_boxes = []

        for frame_num, player_bbox in enumerate(player_boxes):
            ball_box = ball_boxes[frame_num][1]
            ball_position = get_center_of_bbox(ball_box)
            closest_player_id_to_ball = min(
                player_bbox.keys(),
                # 根据球和球员之间的距离来比较
                key=lambda x: measure_distance(
                    ball_position, get_center_of_bbox(player_bbox[x])
                ),
            )

            output_player_bboxes_dict = {}
            for player_id, bbox in player_bbox.items():
                foot_position = get_foot_position(bbox)

                # 计算离球员最近的关键点
                closest_key_point_index = get_closest_keypoint_index(
                    foot_position, original_court_key_points, [0, 2, 12, 13]
                )
                closest_key_point = (
                    original_court_key_points[closest_key_point_index * 2],
                    original_court_key_points[closest_key_point_index * 2 + 1],
                )

                # 计算球员的最大高度（防止球员因下蹲等动作导致的高度变化）
                frame_index_min = max(0, frame_num - 20)
                frame_index_max = min(len(player_boxes), frame_num + 50)
                bboxes_heights_in_pixels = [
                    get_height_of_bbox(player_boxes[i][player_id])
                    for i in range(frame_index_min, frame_index_max)
                ]
                max_player_height_in_pixels = max(bboxes_heights_in_pixels)

                # 计算球员在 mini_court 上的位置
                mini_court_player_position = self.get_mini_court_coordinates(
                    foot_position,
                    closest_key_point,
                    closest_key_point_index,
                    max_player_height_in_pixels,
                    player_heights[player_id],
                )
                output_player_bboxes_dict[player_id] = mini_court_player_position

                # 如果球员是离球最近的球员，才计算球在 mini_court 上的位置，这样可以在 mini_court 上较为准确地表示球的位置。
                if closest_player_id_to_ball == player_id:
                    # 计算离球最近的关键点
                    closest_key_point_index = get_closest_keypoint_index(
                        ball_position, original_court_key_points, [0, 2, 12, 13]
                    )
                    closest_key_point = (
                        original_court_key_points[closest_key_point_index * 2],
                        original_court_key_points[closest_key_point_index * 2 + 1],
                    )

                    mini_court_player_position = self.get_mini_court_coordinates(
                        ball_position,
                        closest_key_point,
                        closest_key_point_index,
                        max_player_height_in_pixels,
                        player_heights[player_id],
                    )

                    output_ball_boxes.append({1: mini_court_player_position})
            output_player_boxes.append(output_player_bboxes_dict)
        return output_player_boxes, output_ball_boxes

    def draw_points_on_mini_court(self, frames, positions, color=(0, 255, 0)):
        for frame_num, frame in enumerate(frames):
            for _, position in positions[frame_num].items():
                x, y = position
                x = int(x)
                y = int(y)
                cv2.circle(frame, (x, y), 5, color, -1)
        return frames
