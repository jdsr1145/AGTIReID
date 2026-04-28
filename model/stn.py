import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# -----------------------------------------------------------
# Helper: solve homography from 4 point pairs using DLT
# source, target: (B,4,2) in normalized [-1,1] coords
# returns H matrices: (B,3,3)
# -----------------------------------------------------------
def solve_homography(src, dst):
    B = src.size(0)
    device = src.device

    # Output tensor
    H = torch.zeros(B, 3, 3, device=device)

    for b in range(B):
        s = src[b]  # (4,2)
        d = dst[b]

        # Build linear system A h = 0
        A = []

        for (x, y), (xp, yp) in zip(s, d):
            A.append([0, 0, 0, -x, -y, -1, yp*x, yp*y, yp])
            A.append([x, y, 1, 0, 0, 0, -xp*x, -xp*y, -xp])

        A = torch.tensor(A, dtype=torch.float32, device=device)  # (8,9)

        # Solve Ah=0 using SVD
        U, S, V = torch.linalg.svd(A)
        h = V[-1]  # last row is smallest singular vector

        H[b] = h.view(3,3)

    return H


# -----------------------------------------------------------
# Localization Network (predicts p1, p2, theta)
# -----------------------------------------------------------
class LocalizationNet(nn.Module):
    def __init__(self, in_channels=3, max_angle=math.pi/2):
        super().__init__()
        self.max_angle = max_angle

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fc = nn.Linear(128, 5)

        with torch.no_grad():
            self.fc.weight.zero_()
            self.fc.bias.copy_(torch.tensor([-0.7, -0.7, 0.7, 0.7, 0.0]))

    def forward(self, x):
        B = x.size(0)
        feat = self.features(x).view(B, -1)
        raw = self.fc(feat)

        pts = torch.tanh(raw[:, :4]) * 0.9
        p1 = pts[:, :2]
        p2 = pts[:, 2:]

        theta = self.max_angle * torch.tanh(raw[:, 4])

        return p1, p2, theta


# -----------------------------------------------------------
# Homography-based STN
# -----------------------------------------------------------
class HomographySTN(nn.Module):
    def __init__(self, in_channels=3, out_h=256, out_w=128):
        super().__init__()
        self.loc_net = LocalizationNet(in_channels)
        self.out_h = out_h
        self.out_w = out_w

        # Canonical output corners in normalized coords
        self.register_buffer("dst_corners", torch.tensor([
            [-1, -1],
            [ 1, -1],
            [ 1,  1],
            [-1,  1]
        ], dtype=torch.float32).unsqueeze(0))  # (1,4,2)

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device

        # 1. Predict parameters
        # TODO: can points go out of image?
        p1, p2, theta = self.loc_net(x)

        # # HACK: params
        # import math
        # # manually define parameters
        # p1 = torch.tensor([[-0.75, -0.75]])   # top-left approx
        # p2 = torch.tensor([[ 0.75,  0.75]])   # bottom-right approx
        # theta = torch.tensor([math.radians(20.0)])

        # 2. Build rectangle from p1, p2
        x_min = torch.min(p1[:,0], p2[:,0])
        x_max = torch.max(p1[:,0], p2[:,0])
        y_min = torch.min(p1[:,1], p2[:,1])
        y_max = torch.max(p1[:,1], p2[:,1])

        cx = (x_min + x_max)/2
        cy = (y_min + y_max)/2
        hw = (x_max - x_min)/2
        hh = (y_max - y_min)/2

        # 3. Corners before rotation
        corners = torch.stack([
            torch.stack([-hw, -hh], dim=-1),
            torch.stack([ hw, -hh], dim=-1),
            torch.stack([ hw,  hh], dim=-1),
            torch.stack([-hw,  hh], dim=-1),
        ], dim=1)  # (B,4,2)

        # 4. Rotate
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        R = torch.stack([
            torch.stack([cos_t, -sin_t], dim=-1),
            torch.stack([sin_t,  cos_t], dim=-1),
        ], dim=1)  # (B,2,2)

        rot = torch.bmm(corners, R.transpose(1,2))   # (B,4,2)
        ctr = torch.stack([cx, cy], dim=1).unsqueeze(1)  # (B,1,2)
        src_corners = rot + ctr  # (B,4,2) in normalized coords

        # 5. Compute homography
        dst_corners = self.dst_corners.expand(B, -1, -1)  # (B,4,2)
        H_mat = solve_homography(dst_corners, src_corners)  # (B,3,3)

        # 6. Generate grid
        # Build normalized grid in output space
        ys, xs = torch.linspace(-1,1,self.out_h,device=device), torch.linspace(-1,1,self.out_w,device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # (H,W)
        grid = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)   # (H,W,3)
        grid = grid.view(1, self.out_h*self.out_w, 3).expand(B, -1, -1)  # (B,HW,3)

        # Apply homography
        warped = torch.bmm(grid, H_mat.transpose(1,2))  # (B,HW,3)
        warped_xy = warped[..., :2] / warped[..., 2:3]

        grid_final = warped_xy.view(B, self.out_h, self.out_w, 2)

        # 7. Sample
        out = F.grid_sample(x, grid_final, align_corners=True)

        return out, {
            "p1": p1,
            "p2": p2,
            "theta": theta,
            "src_corners": src_corners
        }

def build_stn():
    # TODO: optimal target corners?
    in_channels = 3
    out_h = 256
    out_w = 128

    stn = HomographySTN(in_channels=in_channels, out_h=out_h, out_w=out_w)
    return stn