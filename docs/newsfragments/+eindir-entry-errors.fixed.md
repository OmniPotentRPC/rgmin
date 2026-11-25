`rgmin_minimize_eindir` and `rgmin_minimize_eindir_opts` now name the
symbol that was called in their error messages. The options entry used
to report under the legacy name. A failing eindir callback no longer
writes a fixed message on every refused trial: the entry sets the
evaluation failure message after the run, only when the run ends in an
oracle error, so a trial that recovers does not hide a later error.
