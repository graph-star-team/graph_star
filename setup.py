import sys
import platform
import subprocess

# for torch 1.7.0
POSSIBLE_CUDAS = ["cpu", "cu92", "cu101", "cu102", "cu110"]
TORCH_VERIONS = [
    "torch-scatter",
    "torch-sparse",
    "torch-cluster",
    "torch-spline-conv",
    "torch-geometric",
]
TORCH_STABLE = "https://download.pytorch.org/whl/torch_stable.html"


def install_reqs():
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", "requirements2.txt"]
    )


def install_torch(CUDA):
    url = "https://pytorch-geometric.com/whl/torch-1.7.0+" + CUDA + ".html"

    # install base torch
    torch = "torch==1.7.0"
    if CUDA != "cpu":
        torch += "+" + CUDA
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", torch, "-f", TORCH_STABLE]
    )

    # install geometric
    for version in TORCH_VERIONS:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", version, "-f", url]
        )


if __name__ == "__main__":
    file_name = sys.argv[0]
    CUDA = sys.argv[-1]
    OS = platform.system()
    if OS == "Darwin" and CUDA != "cpu":
        print("Running MacOS. Only possible torch is cpu based.")
        CUDA = "cpu"
    if CUDA not in POSSIBLE_CUDAS:
        sys.exit(f"""
                {CUDA} is invalid type of torch 1.7.0.
                Choose from {POSSIBLE_CUDAS}"""
                )
                
    install_torch(CUDA)
    install_reqs()
