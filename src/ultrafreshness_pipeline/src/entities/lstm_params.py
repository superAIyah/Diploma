import dataclasses


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class LstmParams:
    input_dim: int
    hidden_dim: int
    layer_dim: int
    output_dim: int
