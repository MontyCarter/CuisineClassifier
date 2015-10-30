(TeX-add-style-hook "interim"
 (lambda ()
    (LaTeX-add-bibliographies
     "refs")
    (LaTeX-add-labels
     "sec:background"
     "sec:plan")
    (TeX-add-symbols
     "reals")
    (TeX-run-style-hooks
     "url"
     "inputenc"
     "utf8"
     "amssymb"
     "amsthm"
     "amsmath"
     ""
     "geometry"
     "letterpaper"
     "latex2e"
     "art11"
     "article"
     "11pt")))

