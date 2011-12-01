#!/bin/bash

rst2html.py ispc.txt > ispc.html
rst2html.py perf.txt > perf.html
rst2html.py faq.txt > faq.html

#rst2latex --section-numbering --documentclass=article --documentoptions=DIV=9,10pt,letterpaper ispc.txt > ispc.tex
#pdflatex ispc.tex
#/bin/rm -f ispc.aux ispc.log ispc.out ispc.tex
