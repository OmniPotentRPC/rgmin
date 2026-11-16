`Report` gains `grad: Option<Array1<f64>>`, the gradient that `grad_norm` measures.
Sessions fill it and one-shot drivers leave it `None`.
Code that builds a `Report` literal must set the field.
