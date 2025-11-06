"""Reconstruct 3D image from k-space data and trajectory."""

import time
from typing import Optional

import numpy as np
from absl import app, logging

from recon import dcf, kernel, proximity, recon_model, system_model
from recon.cs import convexalg
from recon.cs import sigpy as sp
from recon.cs import tv
from utils import img_utils, io_utils


def reconstruct(
    data: np.ndarray,
    traj: np.ndarray,
    kernel_sharpness: float = 0.32,
    kernel_extent: float = 0.32 * 9,
    overgrid_factor: int = 3,
    image_size: int = 128,
    n_dcf_iter: int = 20,
    verbosity: bool = True,
) -> np.ndarray:
    """Reconstruct k-space data and trajectory.

    Args:
        data (np.ndarray): k space data of shape (K, 1)
        traj (np.Jlndarray): k space trajectory of shape (K, 3)
        kernel_sharpness (float): kernel sharpness. larger kernel sharpness is sharper
            image
        kernel_extent (float): kernel extent.
        overgrid_factor (int): overgridding factor
        image_size (int): target reconstructed image size
            (image_size, image_size, image_size)
        n_pipe_iter (int): number of dcf iterations
        verbosity (bool): Log output messages

    Returns:
        np.ndarray: reconstructed image volume
    """
    start_time = time.time()
    prox_obj = proximity.L2Proximity(
        kernel_obj=kernel.Gaussian(
            kernel_extent=kernel_extent,
            kernel_sigma=kernel_sharpness,
            verbosity=verbosity,
        ),
        verbosity=verbosity,
    )
    system_obj = system_model.MatrixSystemModel(
        proximity_obj=prox_obj,
        overgrid_factor=overgrid_factor,
        image_size=np.array([image_size, image_size, image_size]),
        traj=traj,
        verbosity=verbosity,
    )
    dcf_obj = dcf.IterativeDCF(
        system_obj=system_obj, dcf_iterations=n_dcf_iter, verbosity=verbosity
    )
    recon_obj = recon_model.LSQgridded(
        system_obj=system_obj, dcf_obj=dcf_obj, verbosity=verbosity
    )
    image = recon_obj.reconstruct(data=data, traj=traj)
    del recon_obj, dcf_obj, system_obj, prox_obj
    end_time = time.time()
    execution_time = end_time - start_time
    logging.info("Execution time: {:.2f} seconds".format(execution_time))
    return image


def reconstruct_cs(
    data: np.ndarray,
    traj: np.ndarray,
    image_size: int,
    overgrid_factor: int = 1,
    k: Optional[np.ndarray] = None,
    LL: Optional[float] = None,
) -> np.ndarray:
    """Reconstruct using compressed sensing on invivo data.

    Args:
        data (np.ndarray): k space data of shape (K, 1)
        traj (np.ndarray): k space trajectory of shape (K, 3)
        image_size (int): target reconstructed image size
        k (np.np.ndarray): k-space decay matrix of size (K, 1)
        LL (float): maximum eigenvalue of A^H A
    """
    start_time = time.time()
    # set constants
    num_iters = 33
    lamda_2 = 3e-5
    lamda_1 = 1e-6
    rho = 1e2
    ptol = 1e-3
    # num_normal = 30
    num_normal = 8
    # set device
    devnum = 0
    device = sp.Device(devnum)
    if devnum == -1:
        device = sp.cpu_device
    xp = device.xp
    # create sensitivity map
    sense_map = np.ones(
        (
            1,
            overgrid_factor * image_size,
            overgrid_factor * image_size,
            overgrid_factor * image_size,
        ),
        dtype=int,
    )

    if k is None:
        k = np.ones_like(data)

    # reshape and normalize inputs
    data = np.conjugate(data.reshape((1, 1, -1)))
    k = k.reshape((1, 1, -1))
    traj = traj.reshape((1, -1, 3)) * overgrid_factor * image_size
    with device:
        # move data to device
        sense_map = sp.to_device(sense_map, device=device)
        data = sp.to_device(data, device=device)
        traj = sp.to_device(traj, device=device)
        # compute linear operators
        S = sp.linop.Multiply(
            (
                overgrid_factor * image_size,
                overgrid_factor * image_size,
                overgrid_factor * image_size,
            ),
            sense_map,
        )
        F = sp.linop.NUFFT(
            sense_map.shape, coord=traj, oversamp=1.5, width=4, toeplitz=True
        )

        K = sp.linop.Multiply(F.oshape, k)
        A = K * F * S
        # normalize by maximum eigenvalue
        if LL is None:
            LL = sp.app.MaxEig(A.N, dtype=xp.complex64, device=device).run() * 1.01
        A = np.sqrt(1 / LL) * A
        # breakpoint()
        # define regularizing linear operators and their proximal operators
        W = sp.linop.Wavelet(S.ishape, wave_name="db4")
        prox_g1 = sp.prox.UnitaryTransform(sp.prox.L1Reg(W.oshape, lamda_1), W)
        prox_g2 = tv.ProxTV(A.ishape, lamda_2)
        # make list of objectives and proximal operators
        list_g = [
            lambda x: lamda_1 * xp.linalg.norm(W(x).ravel(), ord=1),
            lambda x: lamda_2 * xp.linalg.norm(prox_g2.G(x)),
        ]
        list_proxg = [prox_g1, prox_g2]
        # reconstruction using ADMM
        image = sp.to_device(
            convexalg.admm(
                num_iters=num_iters,
                ptol=ptol,
                A=A,
                b=data,
                num_normal=num_normal,
                lst_proxg=list_proxg,
                rho=rho,
                lst_g=list_g,
                method="cg",
                verbose=True,
                draw_output=False,
            ),
            sp.cpu_device,
        )
    if overgrid_factor > 1:
        image = img_utils.crop_center(image, image_size)
    end_time = time.time()
    logging.info("Execution time: {:.2f} seconds".format(end_time - start_time))
    image *= np.sqrt(1 / LL)
    return image


def main(argv):
    """Demonstrate non-cartesian reconstruction.

    Uses demo data from the assets folder.
    """
    data = io_utils.import_mat("assets/demo_radial_mri_data.mat")["data"]
    traj = io_utils.import_mat("assets/demo_radial_mri_traj.mat")["traj"]
    image = reconstruct(
        data=data,
        traj=traj,
        kernel_sharpness=1.0 / 3,
        kernel_extent=2,
        n_dcf_iter=10,
        verbosity=True,
    )
    image = img_utils.flip_and_rotate_image(image)
    io_utils.export_nii(np.abs(image), "tmp/demo_nufft.nii")

    logging.info("done!")


if __name__ == "__main__":
    app.run(main)
