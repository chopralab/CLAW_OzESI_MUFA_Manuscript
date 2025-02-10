import torch
import ast
import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--f1", action="store_true", default=False)
parser.add_argument("--f2", action="store_true", default=False)
parser.add_argument("--vector", type=str, required=True)
args = parser.parse_args()

def function_1(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(
        -10 * torch.exp(
            -0.2 * torch.sqrt(x[0:2] ** 2.0 + x[1:3] ** 2.0)
        ),
        dim=-1,
    )

def function_2(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(
        (torch.abs(x) ** 0.8) + (5 * torch.sin(x ** 3)),
        dim=-1,
    )

if __name__ == "__main__":
    vector = ast.literal_eval(args.vector)

    output = {}

    if args.f1: output["f1"] = float(function_1(torch.tensor(vector)))
    if args.f2: output["f2"] = float(function_2(torch.tensor(vector)))

    print(json.dumps(output, indent=2))