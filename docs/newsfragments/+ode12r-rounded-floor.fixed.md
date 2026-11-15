An ODE12r session step that no cap or bound shortened is judged at its own length `h`, and the next trial starts from the length the controller chose.
Before, the session recomputed the taken length as `h |x' - x| / |dx|` for every move, and that ratio rounds to a few ulps below `h` about as often as not.
An accepted move that rounded short then took the capped-move floor, so the next trial reused `h` in place of the shorter line-search or error-controlled length.
Whether a step was floored depended on rounding alone: a change in the last bits of the forces, such as the libm Exp weight, moved the step count of a run.
A capped or clipped move is judged and floored as before.
