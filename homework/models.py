from pathlib import Path
import torch
import torch.nn as nn


HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class ConvBlock(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int=0,
        stride: int=1,
        groups: int=1
    ):
        
        super().__init__()

        layers = [
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                groups=groups
            ),
            torch.nn.BatchNorm2d(
                out_channels
            ),
            torch.nn.ReLU()
        ]

        self.block = torch.nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        include_dropout: bool=False
    ):
        
        super().__init__()
        
        layers = [
            ConvBlock(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            ConvBlock(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            )
        ]

        if include_dropout:
            layers.append(torch.nn.Dropout(p=.2))

        self.block = torch.nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleBlock(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        include_dropout: bool=False
    ):
        
        super().__init__()

        layers = [
            torch.nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                output_padding=1,
                stride=2
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        ]

        if include_dropout:
            layers.append(
                torch.nn.Dropout(p=.2)
            )

        self.block = torch.nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.block(x)


class RESBlock(nn.Module):

    def __init__(self, in_size: int, out_size: int):

        super().__init__()

        trans_layers = [
            nn.Linear(in_size, out_size, bias=False),
            nn.LayerNorm(out_size),
            nn.GELU()
        ]

        self.trans = nn.Sequential(*trans_layers)
        self.skip = nn.Identity()

        if in_size != out_size:
            self.skip = nn.Linear(in_size, out_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        trans_output = self.trans(x)
        skip_output = self.skip(x)

        return trans_output + skip_output

class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        layer_sizes: list=[1164, 100, 50, 10],
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        layers = []

        o = self.n_track * 4
        for layer in layer_sizes:

            layers.append(RESBlock(o, layer))
            o = layer
        
        layers.append(torch.nn.Linear(o, self.n_waypoints * 2))
        self.model = torch.nn.Sequential(*layers)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """

        track_left_flat = track_left.reshape(-1, self.n_track * 2)
        track_right_flat = track_right.reshape(-1, self.n_track * 2)
        x = torch.concat((track_left_flat, track_right_flat), dim=1)

        # normalize the inputs
        x = (x - x.mean(dim=1)[..., None]) / torch.std(x, dim=1)[..., None]
        y = self.model(x)

        return y.reshape(-1, self.n_waypoints, 2)


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.query_embed = nn.Embedding(n_waypoints, d_model)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        raise NotImplementedError


class CNNPlanner(torch.nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        n_waypoints: int=3
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # first conv layer
        self.first_layer = ConvBlock(
            in_channels,
            32,
            kernel_size=3,
            padding=1
        )

        # encoding layers
        self.encoder_layers = torch.nn.ModuleList()
        current_output_size = 32
        encoding_layers = 3

        for layer in range(encoding_layers):

            include_dropout = False

            if layer == (encoding_layers - 1):
                include_dropout = True

            self.encoder_layers.append(
                EncoderBlock(
                    current_output_size,
                    current_output_size * 2,
                    include_dropout=include_dropout
                )
            )

            current_output_size *= 2

        
        # bottleneck layers
        self.bottleneck_layers = [
            ConvBlock(
                current_output_size,
                current_output_size * 2,
                kernel_size=3,
                padding=1
            ),
            torch.nn.Dropout(p=.5),
            ConvBlock(
                current_output_size * 2,
                current_output_size,
                kernel_size=3,
                padding=1
            )
        ]

        self.bottleneck = torch.nn.Sequential(*self.bottleneck_layers)
        
        # define decode layers
        self.decode_layers = torch.nn.ModuleList()
        for i, _ in enumerate(self.encoder_layers):

            include_dropout = False

            if i == (encoding_layers - 1):
                include_dropout = True

            first_decode_layer = ConvBlock(
                current_output_size,
                current_output_size,
                kernel_size=3,
                padding=1
            )

            upsample_layer = UpsampleBlock(
                current_output_size * 2,
                current_output_size // 2,
                include_dropout
            )

            decode_package = torch.nn.ModuleList([first_decode_layer, upsample_layer])
            self.decode_layers.append(decode_package)

            current_output_size //= 2

        self.n_waypoints = n_waypoints

        # output transformations
        self.waypoint_head = torch.nn.Sequential(
            ConvBlock(
                current_output_size,
                1280,
                kernel_size=1
            ),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(1280, self.n_waypoints * 2, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # first full convolution
        current_map = self.first_layer(z)
        
        # encoder layers
        encoder_feature_maps = list()
        for encoder in self.encoder_layers:

            encoder_feature_map = encoder(current_map)

            # save these for res connections between decoder and encoder
            encoder_feature_maps.append(encoder_feature_map)
            current_map = encoder_feature_map
        
        # bottleneck
        x = self.bottleneck(current_map)

        zip_decode_iter = zip(
            self.decode_layers,
            reversed(encoder_feature_maps) # iterate through encoder map stack
        )
        # decode layers
        for (fl, upsample), encoder_feature_map in zip_decode_iter:
            
            # concat the first layer with the encoder map
            x = torch.concat([fl(x), encoder_feature_map], dim=1)

            # upsample the concatenation
            x = upsample(x)

        # output transformation heads
        waypoints = self.waypoint_head(x).reshape(-1, self.n_waypoints, 2)

        return waypoints


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
