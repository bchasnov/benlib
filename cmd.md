## Useful commands

* Invert color of PDF figures for presentations
`parallel convert -density 300 {} -channel RGB -negate -density 100 inverted_{} ::: *.pdf`
