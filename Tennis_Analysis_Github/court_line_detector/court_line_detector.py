import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import pprint


class CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14 * 2)
        self.model.load_state_dict(torch.load(model_path))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.transform = transforms.Compose(
            [
                # OpenCV 读取的图像是 NumPy 数组格式，而许多 PyTorch 的图像变换操作需要 PIL 图像对象作为输入。
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    # 预测 image 上的 14 * 2 个关键点坐标
    def predict(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # unsqueeze(0) 在第 0 维插入一个大小为 1 的新维度。如果 self.transform(img_rgb) 的输出形状是 (C, H, W)，那么 unsqueeze(0) 后的形状将变为 (1, C, H, W)。
        # 使用原因：深度学习模型通常期望输入数据是批处理的形式，即使只输入一张图像。批处理的张量形状通常是 (N, C, H, W)，其中 N 是批处理的大小（batch size）。
        image_tensor = self.transform(img_rgb).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        # 在 GPU 上的张量直接转换为 NumPy 数组是不行的，需要先将张量从 GPU 移动到 CPU，然后再转换为 NumPy 数组。
        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = img_rgb.shape[:2]

        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0

        return keypoints

    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i + 1])

            cv2.putText(
                image,
                str(i // 2),
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 225),
                2,
            )
            # -1 表示填充整个圆。如果是正数，则表示圆的边框厚度。
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image

    def draw_keypoints_on_video(self, vido_frames, keypoints):
        output_video_frames = []
        for frame in vido_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames
