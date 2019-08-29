#!/bin/bash

pdflatex -synctex=1 -interaction=nonstopmode thesis.tex
rm thesis.bbl
bibtex thesis
pdflatex -synctex=1 -interaction=nonstopmode thesis.tex
pdflatex -synctex=1 -interaction=nonstopmode thesis.tex
rm thesis.aux
rm thesis.blg
rm thesis.dvi
rm thesis.out
rm thesis.thm
rm thesis.toc
# rm thesis.synctex.gz
