;; The docs runner's Emacs ships Org 9.6. ox-rst calls
;; org-element-type-p, which arrived in Org 9.7, so install a current
;; Org from GNU ELPA before ox-rst is byte-compiled against it.
(require 'package)
(add-to-list 'package-archives '("gnu" . "https://elpa.gnu.org/packages/") t)
(add-to-list 'package-archives '("melpa" . "https://melpa.org/packages/") t)
(setq package-install-upgrade-built-in t)
(package-initialize)

(defconst rgmin-docs-org-min '(9 7)
  "Oldest Org release that defines org-element-type-p.")

(unless (package-installed-p 'org rgmin-docs-org-min)
  (package-refresh-contents)
  (let ((desc (cadr (assq 'org package-archive-contents))))
    (unless desc
      (error "Org is missing from GNU ELPA"))
    (unless (version-list-<= rgmin-docs-org-min (package-desc-version desc))
      (error "GNU ELPA Org %s is older than 9.7"
             (package-version-join (package-desc-version desc))))
    (package-install desc)))

;; Prefer the ELPA release over the built-in Org on load-path.
(let* ((descs (cdr (assq 'org package-alist)))
       (dir (and descs (package-desc-dir (car descs)))))
  (when (and (stringp dir) (file-directory-p dir))
    (setq load-path (cons dir (delete dir load-path)))))

(require 'org-element)
(unless (fboundp 'org-element-type-p)
  (error "Org %s has no org-element-type-p" (org-version)))
(message "Using Org %s" (org-version))

(unless (package-installed-p 'ox-rst)
  (unless package-archive-contents
    (package-refresh-contents))
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
