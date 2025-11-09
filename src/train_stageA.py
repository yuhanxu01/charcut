import argparse, os, torch, torchvision
from torch.utils.data import DataLoader
from dataset import ParagraphDataset
from models.build_maskrcnn import build_maskrcnn
from tqdm import tqdm

def collate_fn(batch):
    imgs, tgts = zip(*batch)
    return list(imgs), list(tgts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="runs/stageA")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ds = ParagraphDataset(args.data)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_maskrcnn(num_classes=2, pretrained=True).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)

    for ep in range(args.epochs):
        model.train()
        pbar = tqdm(dl, desc=f"ep{ep}")
        for imgs, targets in pbar:
            imgs = [torch.from_numpy(torchvision.transforms.functional.pil_to_tensor(x).numpy()/255.).float().to(device) for x in imgs]
            # convert to list of tensors already ok
            targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k,v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix({k: float(v.detach().cpu()) for k,v in loss_dict.items()})
        torch.save(model.state_dict(), os.path.join(args.out, f"epoch{ep}.pt"))
    torch.save(model.state_dict(), os.path.join(args.out, "last.pt"))

if __name__ == "__main__":
    main()
