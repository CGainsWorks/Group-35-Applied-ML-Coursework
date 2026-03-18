import cv2
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from scipy.io import loadmat


class JHMDBDataset(Dataset):
    def __init__(self, root, split="train", split_id=1, num_frames=8, transform=None):
        self.root = Path(root)
        self.split = split
        self.split_id = split_id
        self.num_frames = num_frames
        self.transform = transform

        self.video_root = self.root / "ReCompress_Videos"
        self.joint_root = self.root / "joint_positions"
        self.split_root = self.root / "splits"

        self.classes = sorted([
            d.name for d in self.video_root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        self._discover()

    def _discover(self):
        for class_name in self.classes:
            split_file = self.split_root / f"{class_name}_test_split{self.split_id}.txt"
            if not split_file.exists():
                continue

            with open(split_file, "r") as f:
                lines = [line.strip() for line in f if line.strip()]

            for line in lines:
                parts = line.split()
                if len(parts) < 2:
                    continue

                video_name = parts[0]
                split_flag = int(parts[1])

                if self.split == "train" and split_flag != 1:
                    continue
                if self.split == "test" and split_flag != 2:
                    continue

                video_path = self.video_root / class_name / video_name
                video_stem = Path(video_name).stem
                joint_path = self.joint_root / class_name / video_stem / "joint_positions.mat"

                if not video_path.exists():
                    continue
                if not joint_path.exists():
                    continue

                self.samples.append({
                    "video_path": video_path,
                    "joint_path": joint_path,
                    "label": self.class_to_idx[class_name],
                    "class_name": class_name,
                    "video_name": video_name,
                })

    def __len__(self):
        return len(self.samples)

    def _sample_indices(self, total_frames):
        step = total_frames / self.num_frames
        indices = []
        for i in range(self.num_frames):
            idx = int(i * step)
            idx = min(idx, total_frames - 1)
            indices.append(idx)
        return indices

    def _read_video_frames(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames found in {video_path}")

        return frames

    def _joints_to_box(self, joints_xy, pad=10):
        joints_xy = joints_xy[~np.isnan(joints_xy).any(axis=1)]

        if len(joints_xy) == 0:
            return [0, 0, 0, 0]
        xs = joints_xy[:, 0]
        ys = joints_xy[:, 1]

        x1 = max(0, float(xs.min() - pad))
        y1 = max(0, float(ys.min() - pad))
        x2 = float(xs.max() + pad)
        y2 = float(ys.max() + pad)

        return [x1, y1, x2, y2]

    def _load_joint_boxes(self, joint_path, sampled_indices):
        mat = loadmat(joint_path)

        possible_keys = [k for k in mat.keys() if not k.startswith("__")]

        if "pos_img" in mat:
            joints = mat["pos_img"]
        else:
            # fallback: first non-metadata key
            joints = mat[possible_keys[0]]

        joints = np.array(joints)

        boxes = []

        for idx in sampled_indices:
            idx = min(idx, joints.shape[-1] - 1)
            if joints.ndim == 3 and joints.shape[0] == 2:
                # shape: (2, J, T)
                x = joints[0, :, idx]
                y = joints[1, :, idx]
                joints_xy = np.stack([x, y], axis=1)

            elif joints.ndim == 3 and joints.shape[1] == 2:
                # shape: (J, 2, T)
                joints_xy = joints[:, :, idx]

            else:
                raise ValueError(f"Unexpected joint shape {joints.shape} in {joint_path}")

            box = self._joints_to_box(joints_xy)
            boxes.append(box)

        return torch.tensor(boxes, dtype=torch.float32)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        frames = self._read_video_frames(sample["video_path"])
        sampled_indices = self._sample_indices(len(frames))
        sampled_frames = [frames[i] for i in sampled_indices]

        pil_frames = [T.ToPILImage()(f) for f in sampled_frames]

        if self.transform:
            frames_tensor = torch.stack([self.transform(img) for img in pil_frames])
        else:
            frames_tensor = torch.stack([T.ToTensor()(img) for img in pil_frames])

        boxes = self._load_joint_boxes(sample["joint_path"], sampled_indices)
        h, w = sampled_frames[0].shape[:2]
        boxes[:, [0, 2]] /= w
        boxes[:, [1, 3]] /= h
        label = torch.tensor(sample["label"], dtype=torch.long)

        meta = {
            "class_name": sample["class_name"],
            "video_name": sample["video_name"],
        }

        return frames_tensor, boxes, label, meta

if __name__ == "__main__":
    import torchvision.transforms as T

    dataset = JHMDBDataset(
        root="./JHMDB",
        split="train",
        split_id=1,
        num_frames=8,
        transform=T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])
    )

    print("Samples:", len(dataset))

    frames, boxes, label, meta = dataset[0]

    print("Frames:", frames.shape)
    print("Boxes:", boxes.shape)
    print("Label:", label)
    print("Meta:", meta)