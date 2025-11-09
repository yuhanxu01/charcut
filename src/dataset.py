import os, json, numpy as np, torch
from PIL import Image

class ParagraphDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split="train"):
        ann_path = os.path.join(root_dir, "annotations.json")
        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.root = root_dir
        self.images = data["images"]
        self.ann = data["annotations"]
        # build idx -> instances
        self.by_image = {}
        for a in self.ann:
            self.by_image.setdefault(a["image_id"], []).append(a)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = self.images[idx]
        img_path = os.path.join(self.root, item["file_name"])
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        insts = self.by_image.get(item["id"], [])
        masks = []
        boxes = []
        labels = []
        for a in insts:
            mpath = os.path.join(self.root, a["mask_path"])
            m = Image.open(mpath).convert("L")
            m_np = (np.array(m) > 127).astype("uint8")
            ys, xs = np.where(m_np > 0)
            if xs.size == 0 or ys.size == 0:
                continue
            x0, y0 = xs.min(), ys.min()
            x1, y1 = xs.max()+1, ys.max()+1
            boxes.append([x0, y0, x1, y1])
            masks.append(m_np)
            labels.append(1)  # class-agnostic
        if len(boxes) == 0:
            boxes = np.zeros((0,4), dtype=np.float32)
            masks_t = torch.zeros((0, H, W), dtype=torch.uint8)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = np.asarray(boxes, dtype=np.float32)
            masks_t = torch.from_numpy(np.stack(masks)).to(torch.uint8)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": labels_t,
            "masks": masks_t,
            "image_id": torch.tensor([item["id"]], dtype=torch.int64),
        }
        return img, target
