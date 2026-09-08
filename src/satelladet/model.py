import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p: int | None = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SatellaBlock(nn.Module):
    """Lightweight split-transform-fuse residual block."""

    def __init__(self, channels: int):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SatellaBlock requires an even channel count.")

        hidden = channels // 2
        self.pre = ConvBNAct(channels, channels, 1, 1)
        self.branch1 = ConvBNAct(hidden, hidden, 3, 1)
        self.branch2 = ConvBNAct(hidden, hidden, 3, 1)
        self.fuse = ConvBNAct(channels, channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pre(x)
        keep, work = y.chunk(2, dim=1)
        work = self.branch2(self.branch1(work))
        return x + self.fuse(torch.cat((keep, work), dim=1))


class SatellaFuse(nn.Module):
    def __init__(self, c_in: int, c_out: int, blocks: int = 1):
        super().__init__()
        self.reduce = ConvBNAct(c_in, c_out, 1, 1)
        self.blocks = nn.Sequential(*[SatellaBlock(c_out) for _ in range(blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.reduce(x))


class SatellaBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvBNAct(3, 24, 3, 2)          # input -> input/2
        self.down2 = ConvBNAct(24, 48, 3, 2)        # input/2 -> input/4
        self.stage2 = nn.Sequential(SatellaBlock(48), SatellaBlock(48))

        self.down3 = ConvBNAct(48, 96, 3, 2)        # input/4 -> input/8
        self.stage3 = nn.Sequential(*[SatellaBlock(96) for _ in range(3)])

        self.down4 = ConvBNAct(96, 192, 3, 2)       # input/8 -> input/16
        self.stage4 = nn.Sequential(*[SatellaBlock(192) for _ in range(3)])

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        p2 = self.stage2(self.down2(x))
        p3 = self.stage3(self.down3(p2))
        p4 = self.stage4(self.down4(p3))
        return p2, p3, p4


class SatellaNeck(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce4 = ConvBNAct(192, 96, 1, 1)
        self.fuse3 = SatellaFuse(192, 96)

        self.reduce3 = ConvBNAct(96, 48, 1, 1)
        self.fuse2 = SatellaFuse(96, 48)

        self.down3 = ConvBNAct(48, 96, 3, 2)
        self.out3 = SatellaFuse(192, 96)

        self.down4 = ConvBNAct(96, 192, 3, 2)
        self.out4 = SatellaFuse(384, 192)

    def forward(self, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor):
        x4 = self.reduce4(p4)
        x3 = self.fuse3(
            torch.cat((F.interpolate(x4, scale_factor=2, mode="nearest"), p3), dim=1)
        )
        x2 = self.fuse2(
            torch.cat((F.interpolate(self.reduce3(x3), scale_factor=2, mode="nearest"), p2), dim=1)
        )

        o3 = self.out3(torch.cat((self.down3(x2), x3), dim=1))
        o4 = self.out4(torch.cat((self.down4(o3), p4), dim=1))
        return x2, o3, o4


class SatellaHead(nn.Module):
    def __init__(self, num_classes: int = 1, head_channels: int = 64):
        super().__init__()
        self.num_classes = num_classes

        self.projections = nn.ModuleList(
            [
                ConvBNAct(48, head_channels, 1, 1),
                ConvBNAct(96, head_channels, 1, 1),
                ConvBNAct(192, head_channels, 1, 1),
            ]
        )

        # Shared towers across P2/P3/P4 after projection to 64 channels.
        self.box_tower = nn.Sequential(
            ConvBNAct(head_channels, head_channels, 3, 1),
            ConvBNAct(head_channels, head_channels, 3, 1),
        )
        self.cls_tower = nn.Sequential(
            ConvBNAct(head_channels, head_channels, 3, 1),
            ConvBNAct(head_channels, head_channels, 3, 1),
        )

        self.box_pred = nn.Conv2d(head_channels, 4, 1)
        self.cls_pred = nn.Conv2d(head_channels, num_classes, 1)

        # Start with ~1% class probability at every location.
        prior = 0.01
        nn.init.constant_(self.cls_pred.bias, math.log(prior / (1.0 - prior)))

    def forward(self, features):
        outputs = []
        for projection, feature in zip(self.projections, features):
            z = projection(feature)
            outputs.append(
                {
                    "box": self.box_pred(self.box_tower(z)),
                    "cls": self.cls_pred(self.cls_tower(z)),
                }
            )
        return outputs


class SatellaDetN(nn.Module):
    """SatellaDet-N raw training model."""

    strides = (4, 8, 16)

    def __init__(self, num_classes: int = 1):
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        self.num_classes = num_classes
        self.backbone = SatellaBackbone()
        self.neck = SatellaNeck()
        self.head = SatellaHead(num_classes=num_classes)

    def forward(self, x: torch.Tensor):
        p2, p3, p4 = self.backbone(x)
        features = self.neck(p2, p3, p4)
        return self.head(features)


class SatellaDetONNX(nn.Module):
    """
    Generic ONNX deployment wrapper.

    Converts raw per-scale LTRB + class logits into a single tensor:
      [B, 4 + num_classes, N]
    with channels:
      [cx, cy, w, h, class_probability...]
    and boxes in 0..image_size input pixels.
    """

    def __init__(self, model: SatellaDetN, image_size: int = 640):
        super().__init__()
        self.model = model
        self.image_size = float(image_size)
        self.strides = model.strides

    def _decode_level(self, box_raw: torch.Tensor, cls_raw: torch.Tensor, stride: int):
        _, _, h, w = box_raw.shape
        dtype = box_raw.dtype
        device = box_raw.device

        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=dtype, device=device),
            torch.arange(w, dtype=dtype, device=device),
            indexing="ij",
        )

        gx = ((xx + 0.5) * stride).unsqueeze(0).unsqueeze(0)
        gy = ((yy + 0.5) * stride).unsqueeze(0).unsqueeze(0)

        # Positive LTRB distances, expressed in input-image pixels.
        distances = F.softplus(box_raw) * stride
        left = distances[:, 0:1]
        top = distances[:, 1:2]
        right = distances[:, 2:3]
        bottom = distances[:, 3:4]

        x1 = torch.clamp(gx - left, 0.0, self.image_size)
        y1 = torch.clamp(gy - top, 0.0, self.image_size)
        x2 = torch.clamp(gx + right, 0.0, self.image_size)
        y2 = torch.clamp(gy + bottom, 0.0, self.image_size)

        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        bw = x2 - x1
        bh = y2 - y1

        boxes = torch.cat((cx, cy, bw, bh), dim=1).flatten(2)
        scores = torch.sigmoid(cls_raw).flatten(2)
        return torch.cat((boxes, scores), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        levels = self.model(x)
        decoded = []
        for level, stride in zip(levels, self.strides):
            decoded.append(self._decode_level(level["box"], level["cls"], stride))
        return torch.cat(decoded, dim=2)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(size: str = "n", num_classes: int = 1) -> nn.Module:
    """Build a SatellaDet model by size name.

    The registry currently contains SatellaDet-N. The factory keeps the public
    API stable as additional sizes are added later.
    """
    key = str(size).lower().replace("satelladet-", "")
    if key != "n":
        raise ValueError(f"Unknown SatellaDet size {size!r}; available: ['n']")
    return SatellaDetN(num_classes=num_classes)
