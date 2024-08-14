from torch import nn
import lightning as L
import torch


class DilatedCNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=3, out_channels=8, kernel_size=10, stride=10, padding=15
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=8, out_channels=16, kernel_size=4, stride=5, dilation=2
            ),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1, dilation=4),
            nn.ReLU(),
            nn.Flatten(start_dim=-2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(640, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        c = self.cnn_layers(x)
        e = self.linear_layers(c)
        return e


class DilatedCNNDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 640),
            nn.ReLU(),
            nn.Unflatten(dim=-1, unflattened_size=(32, -1)),
        )

        self.cnn_decoder = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=32, out_channels=16, kernel_size=1, dilation=4
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                in_channels=16,
                out_channels=8,
                kernel_size=4,
                stride=5,
                dilation=2,
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                in_channels=8, out_channels=3, kernel_size=10, stride=10, padding=10
            ),
        )

    def forward(self, x):
        ld = self.linear_decoder(x)
        return self.cnn_decoder(ld)


""" Cnn Encoder and Decoder with less reductions"""


class BigDilatedCNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=3, out_channels=32, kernel_size=10, stride=10, padding=15
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=32, out_channels=64, kernel_size=4, stride=5, dilation=2
            ),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, dilation=4),
            nn.ReLU(),
            nn.Flatten(start_dim=-2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(2560, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
        )

    def forward(self, x):
        c = self.cnn_layers(x)
        e = self.linear_layers(c)
        return e


class BigDilatedCNNDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2560),
            nn.ReLU(),
            nn.Unflatten(dim=-1, unflattened_size=(128, -1)),
        )

        self.cnn_decoder = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=128, out_channels=64, kernel_size=1, dilation=4
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                in_channels=64,
                out_channels=32,
                kernel_size=4,
                stride=5,
                dilation=2,
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                in_channels=32, out_channels=3, kernel_size=10, stride=10, padding=10
            ),
        )

    def forward(self, x):
        ld = self.linear_decoder(x)
        return self.cnn_decoder(ld)


class XLDilatedCNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=3, out_channels=32, kernel_size=18, stride=1, padding=15
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=32, out_channels=64, kernel_size=9, stride=1, dilation=2
            ),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, dilation=4),
            nn.ReLU(),
            nn.Flatten(start_dim=-2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(126592, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
        )

    def forward(self, x):
        c = self.cnn_layers(x)
        e = self.linear_layers(c)
        return e


class XLDilatedCNNDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_decoder = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 126592),
            nn.ReLU(),
            nn.Unflatten(dim=-1, unflattened_size=(128, -1)),
        )

        self.cnn_decoder = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=128, out_channels=64, kernel_size=3, dilation=4
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                in_channels=64,
                out_channels=32,
                kernel_size=9,
                stride=1,
                dilation=2,
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                in_channels=32, out_channels=3, kernel_size=18, stride=1, padding=15
            ),
        )

    def forward(self, x):
        ld = self.linear_decoder(x)
        return self.cnn_decoder(ld)


class BranchingCNNEncoder(nn.Module):
    def __init__(self, input_channels=3, base_dim=16):

        super().__init__()

        self.local_branch = nn.Sequential(
            nn.Conv1d(input_channels, base_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(base_dim, base_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.mid_branch = nn.Sequential(
            nn.Conv1d(input_channels, base_dim, kernel_size=9, padding=1, dilation=4),
            nn.ReLU(),
            nn.Conv1d(base_dim, base_dim * 2, kernel_size=9, padding=1, dilation=4),
            nn.ReLU(),
        )
        self.long_brach = nn.Sequential(
            nn.Conv1d(input_channels, base_dim, kernel_size=27, padding=1, dilation=8),
            nn.ReLU(),
            nn.Conv1d(base_dim, base_dim * 2, kernel_size=27, padding=1, dilation=8),
            nn.ReLU(),
        )
        self.fusion = nn.Conv1d(base_dim * 2, base_dim * 4, kernel_size=3, stride=1)
        self.self_attention = nn.MultiheadAttention(embed_dim=base_dim * 4, num_heads=4)

    def forward(self, x):
        local_maps = self.local_branch(x)
        mid_maps = self.mid_branch(x)
        long_maps = self.long_brach(x)

        combined_maps = torch.cat([local_maps, mid_maps, long_maps], dim=2)

        fused = self.fusion(combined_maps)
        fused = fused.permute(2, 0, 1)  # reorganize to seq len, batch, channels
        attend, _ = self.self_attention(fused, fused, fused)
        output = attend.permute(1, 2, 0)

        return output


class BranchingCNNDecoder(nn.Module):
    def __init__(self, output_channels=3, base_dim=16):
        super().__init__()

        self.self_attention = nn.MultiheadAttention(embed_dim=base_dim * 4, num_heads=4)
        self.inverse_fusion = nn.Conv1d(
            base_dim * 4, base_dim * 2, kernel_size=3, stride=1
        )

        self.long_branch = nn.Sequential(
            nn.ConvTranspose1d(
                base_dim * 2, base_dim, kernel_size=27, padding=1, dilation=8
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                base_dim, output_channels, kernel_size=27, padding=1, dilation=8
            ),
            nn.ReLU(),
        )
        self.mid_branch = nn.Sequential(
            nn.ConvTranspose1d(
                base_dim * 2, base_dim, kernel_size=9, padding=1, dilation=4
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                base_dim, output_channels, kernel_size=9, padding=1, dilation=4
            ),
            nn.ReLU(),
        )
        self.local_branch = nn.Sequential(
            nn.ConvTranspose1d(
                base_dim * 2, base_dim, kernel_size=3, padding=1, dilation=1
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                base_dim, output_channels, kernel_size=3, padding=1, dilation=1
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        # permute for attenion block - #(seq len, batch, channels)
        x = x.permute(2, 0, 1)
        x, _ = self.self_attention(x, x, x)

        # permute for  inverse fusion and cnn transpose
        # batch channels, seq leng
        x = x.permute(1, 2, 0)
        x = self.inverse_fusion(x)
        # splits omn sequence leng into 3 dim - recall this is extra long seq

        long_x, mid_x, local_x, _ = torch.split(x, x.size(2) // 3, dim=2)
        long_x = self.long_branch(long_x)
        mid_x = self.mid_branch(mid_x)
        local_x = self.local_branch(local_x)

        long_x = nn.functional.interpolate(
            long_x, size=1000, mode="linear", align_corners=False
        )
        mid_x = nn.functional.interpolate(
            mid_x, size=1000, mode="linear", align_corners=False
        )
        local_x = nn.functional.interpolate(
            local_x, size=1000, mode="linear", align_corners=False
        )

        output = local_x + mid_x + long_x
        return output
