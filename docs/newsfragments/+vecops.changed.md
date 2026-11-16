Every solver's length-n algebra flows through the `vecops` seam.
The seam carries a DLPack-device-tagged `Vector` handle.
Construction refuses a device tag without a kernel backend and never stages it through the host.
