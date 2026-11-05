Links between docs pages and links to the Lean proofs open on the built
site.
They were `file:` links, which the export turns into links to `.rst`
files that the site does not serve.
Links between pages are now `:doc:` roles, and the Lean links go to the
files on GitHub.
