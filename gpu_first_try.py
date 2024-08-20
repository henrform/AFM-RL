from ppafm.io import loadXYZ
from ppafm.ocl.AFMulator import AFMulator
from ppafm.ocl.oclUtils import init_env

# Initialize an OpenCL environment. You can change i_platform to select the device to use.
init_env(i_platform=0)

# Load sample coordinates (xyzs), atomic numbers (Zs), and charges (qs)
xyzs, Zs, qs, _ = loadXYZ("materials/pt_111_angled.xyz")

# Create an instance of the simulator
afmulator = AFMulator(
    scan_dim=(201, 201, 51),
    scan_window=((15.0, 20.0, 26.8), (50.0, 55.0, 30)),
    iZPP=8,
    df_steps=1,
    tipR0=[0.0, 0.0, 3.0]
)

# Run the simulation and plot the resulting images
afm_images = afmulator(xyzs, Zs, qs, plot_to_dir="output")
afmulator.saveFF()