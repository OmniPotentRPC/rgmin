Dogleg updates its trust radius from the dogleg step.
Its Euclidean ceiling is `maxmove` rather than a per-atom displacement cap.
A cap still limits the accepted trial.
Nocedal and Wright, algorithm 4.1.
The 0.2 per-atom default added in 0.4.1 used that cap as the ceiling and fed the clipped length to the boundary test, so the radius stayed near its initial value on Morse bulk.
