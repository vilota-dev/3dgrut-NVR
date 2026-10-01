# Repo notes for agents

## Calibration files

`calibration_files/NV180V2-80.121.0230.json` is the NV180V2 unit 80.121.0230
in device format (3 cameras: sockets 1, 2, 3 = camb, camc, camd; camd is the
reference; camb carries Double Sphere in distortionCoeff[5:11] plus a KB4 fit
in intrinsicMatrix + distortionCoeff[0:4]). It was produced from the unit's
basalt-wrapper factory file by the production converter
/opt/vilota/bin/03-prepare_device_calibration_from_basalt.py — see
~/vilota_results/nv180_sim/report.md for the conversion and its verification.
Two fields need explaining:

- `productName` reads "VK180-V2-FFC" because the converter only accepts its
  allowed product names and its VK180-V2-FFC branch maps basalt cameras
  0/1/2 to sockets CAM_B/C/D — exactly the NV180 layout. The renderer's
  `get_cam_name` resolves camera names through its VK180 branch.
- `deviceName` ("80.121.0230") was set by hand after conversion: the
  converter never writes deviceName, and downstream code uses it as the
  frameId/serial.
