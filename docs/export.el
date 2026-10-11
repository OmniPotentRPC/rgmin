;; Setup Package Manager (to fetch ox-rst automatically)
(require 'package)
(add-to-list 'package-archives '("melpa" . "https://melpa.org/packages/") t)
(package-initialize)

(unless package-archive-contents
  (package-refresh-contents))

;; ox-rst calls org-element-type-p, which Org 9.7 added.
;; Emacs 29 ships Org 9.6.15, and package-installed-p counts that built-in,
;; so install the GNU ELPA package when the active Org is older than 9.7.
(unless (package-installed-p 'org '(9 7))
  (package-install (cadr (assq 'org package-archive-contents))))

;; Ensure ox-rst is present
(unless (package-installed-p 'ox-rst)
  (package-install 'ox-rst))

(require 'ox-rst)
(require 'ox-publish)

;; Define the Publishing Project
(setq org-publish-project-alist
      '(("sphinx-rst"
         :base-directory "./orgmode/"
         :base-extension "org"
         :publishing-directory "./source/"
         :publishing-function org-rst-publish-to-rst
         :recursive t
         :headline-levels 4)))

;; Run the publish
(org-publish "sphinx-rst" t)
