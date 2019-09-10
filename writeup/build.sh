#!/bin/bash

latexdocker pdflatex -synctex=1 -interaction=nonstopmode thesis.tex &> /dev/null
rm thesis.bbl
latexdocker bibtex thesis
latexdocker pdflatex -synctex=1 -interaction=nonstopmode thesis.tex &> /dev/null
latexdocker pdflatex -synctex=1 -interaction=nonstopmode thesis.tex
rm thesis.aux
rm thesis.blg
rm thesis.out
rm thesis.thm
rm thesis.toc
