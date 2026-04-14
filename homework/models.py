from pathlib import Path
import torch
import torch.nn as nn


HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class ElasticLayerNorm(torch.nn.Module):
    '''
        This is the same as layer norm. It adapts elastically to the input
        it recieves. This integration is for NAS for linear layers during training
        where sandwhich sampling will be used
    '''

    def __init__(self, max_in_size: int):

        super().__init__()
        
        # no bias will be learned it will proceed the linear layer
        self.max_in_size = max_in_size
        self.gamma = torch.nn.Parameter(torch.ones(max_in_size))
        
    def forward(self, x: torch.Tensor):

        if x.dim() < 2:
            x = x[None, ]

        return torch.nn.functional.layer_norm(
            x, 
            normalized_shape=(x.shape[-1], ),
            weight=self.gamma[: x.shape[-1]],
            bias=None
        )


# create an elastic Linear Layer
class ElasticLinear(torch.nn.Module):
    '''
        This is an elastic linear layer to train multiple subsets
        of architectures at once.
    '''

    def __init__(self, max_n_in: int, max_n_out: int) -> None:

        super().__init__()

        # generate the weights and bias
        weights = torch.nn.Parameter(
            torch.Tensor(max_n_out, max_n_in),
            requires_grad=True
        )

        bias = torch.nn.Parameter(
            torch.Tensor(max_n_out)
        )
        
        # initialize the weights and bias
        self.weights = torch.nn.init.kaiming_uniform_(weights)

        k = 1 / max_n_in
        self.bias = torch.nn.init.uniform_(bias, -k, k)

    def forward(self, x: torch.Tensor, width: int) -> torch.Tensor:

        '''
            passes x through an elastic linear layer
        '''

        if x.dim() < 2:
            x = x[None, ] # add the batch dimension if not exist

        in_size = x.shape[-1]
        
        sub_weights = self.weights[:width, :in_size]
        sub_bias = self.bias[:width]

        return torch.nn.functional.linear(x, sub_weights, sub_bias)


class ElasticMLPBlock(torch.nn.Module):
    '''
        Elastic MLP block for NAS training.
    '''

    def __init__(self, max_input: int, max_output: int, res=False) -> None:

        '''
            max_width: the max width that the linear block can go.
        '''

        super().__init__()

        self.layers = torch.nn.ModuleDict(
            {
                'linear': ElasticLinear(max_input, max_output),
                'norm': ElasticLayerNorm(max_output),
                'relu': torch.nn.ReLU()
            }
        )
        
        self.res = res
        self.skip = torch.nn.Identity()
    
    def forward(self, x: torch.Tensor, width: int):

        y = self.layers['linear'](x, width)
        y = self.layers['norm'](y)
        y = self.layers['relu'](y)

        if self.res:
            return y + self.skip(x)
            
        return y
    
class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        n_layers: int=78,
        max_width: int=256
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        mlp_layers = []
        mlp_layers.append(ElasticMLPBlock(self.n_track * 4, max_width))

        for _ in range(n_layers):
            
            mlp_layers.append(ElasticMLPBlock(max_width, max_width, res=True))

        mlp_layers.append(ElasticMLPBlock(max_width, self.n_waypoints * 2))

        self.layers = torch.nn.ModuleList(mlp_layers)

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

        width = kwargs.get('width', 256)
        track_left_flat = track_left.reshape(-1, self.n_track * 2)
        track_right_flat = track_right.reshape(-1, self.n_track * 2)
        y = torch.concat((track_left_flat, track_right_flat), dim=1)

        for layer in self.layers:

            y = layer(y, width)

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
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        raise NotImplementedError


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
