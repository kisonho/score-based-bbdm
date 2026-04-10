import torch
from diffusion import Manager
from torchmanager import losses
from typing import TypeVar

from .nn import ABridgeModule, BBDMModule, SDEBBDMModule

U = TypeVar("U", bound=torch.nn.Module)
E = TypeVar('E', bound=torch.nn.Module | None)
D = TypeVar('D', bound=torch.nn.Module | None)

__all__ = ["compile", "compile_sde_bbdm", "compile_bbdm"]


def compile(unet: U, /, time_steps: int = 1000, *, encoder: E = None, decoder: D = None) -> Manager[ABridgeModule[U, E, D]]:
    """
    Compile the diffusion manager with given UNet, time steps, encoder, and decoder for the ABridgeModule.

    - Parameters:
        - unet: The UNet model in `U` to be used in the diffusion process.
        - time_steps: The number of time steps in `int` for the diffusion process.
        - encoder: An optional encoder module in `E` to be used in the diffusion process.
        - decoder: An optional decoder module in `D` to be used in the diffusion process.
    - Returns: A `Manager` instance that manages the diffusion process with the compiled model.
    """
    model = ABridgeModule(unet, time_steps, encoder=encoder, decoder=decoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = losses.MAE()
    return Manager(model, optimizer=optimizer, loss_fn=loss_fn)


def compile_sde_bbdm(unet: U, /, time_steps: int = 1000, *, encoder: E = None, decoder: D = None) -> Manager[SDEBBDMModule[U, E, D]]:
    """
    Compile the diffusion manager with given UNet, time steps, encoder, and decoder for the SDE-based BBDM.

    - Parameters:
        - unet: The UNet model in `U` to be used in the diffusion process.
        - time_steps: The number of time steps in `int` for the diffusion process.
        - encoder: An optional encoder module in `E` to be used in the diffusion process.
        - decoder: An optional decoder module in `D` to be used in the diffusion process.
    - Returns: A `Manager` instance that manages the diffusion process with the compiled model.
    """
    model = SDEBBDMModule(unet, time_steps, encoder=encoder, decoder=decoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = losses.MAE()
    return Manager(model, optimizer=optimizer, loss_fn=loss_fn)


def compile_bbdm(unet: U, /, time_steps: int = 1000, *, encoder: E = None, decoder: D = None) -> Manager[BBDMModule[U, E, D]]:
    """
    Compile the diffusion manager with given UNet, time steps, encoder, and decoder for the traditional BBDM.

    - Parameters:
        - unet: The UNet model in `U` to be used in the diffusion process.
        - time_steps: The number of time steps in `int` for the diffusion process.
        - encoder: An optional encoder module in `E` to be used in the diffusion process.
        - decoder: An optional decoder module in `D` to be used in the diffusion process.
    - Returns: A `Manager` instance that manages the diffusion process with the compiled model.
    """
    model = BBDMModule(unet, time_steps, encoder=encoder, decoder=decoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = losses.MAE()
    return Manager(model, optimizer=optimizer, loss_fn=loss_fn)
