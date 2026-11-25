`StopUnmetError.report` is now owned by the exception. The binding moves
the report into the exception it raises, the class carries a default
`report`, and the stub declares the attribute, so an instance built
directly as `rgmin.StopUnmetError("msg")` has a `report` instead of
raising `AttributeError`.
