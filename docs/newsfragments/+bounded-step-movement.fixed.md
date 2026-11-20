Session and one-shot minimization now measure movement from the point after
projection onto objective bounds. An out-of-bounds starting point therefore
produces the same termination reason, iteration count, and final coordinates
through both APIs.
