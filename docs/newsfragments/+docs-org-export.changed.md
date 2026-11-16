The docs export installs Org 9.7 or newer from the Emacs Lisp Package Archive (ELPA) of the GNU's Not Unix (GNU) project before ox-rst.
Pull requests run the Org export and the Sphinx build.
The Pages deploy stays on a push to `main` or `workflow_dispatch`.
A pull request has its own concurrency group, so it does not cancel a deploy.
`pages: write` and `id-token: write` belong to the deploy job; the workflow token stays read-only.
