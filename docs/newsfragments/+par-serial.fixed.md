The `par` feature keeps vectors under 65536 elements on the serial path.
At those sizes the rayon reductions measured slower than serial.
