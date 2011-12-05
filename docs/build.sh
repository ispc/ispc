#!/bin/bash

for i in ispc perf faq; do
    rst2html.py --template=template.txt --link-stylesheet \
        --stylesheet-path=css/style.css $i.txt > $i.html
done

#rst2latex --section-numbering --documentclass=article --documentoptions=DIV=9,10pt,letterpaper ispc.txt > ispc.tex
#pdflatex ispc.tex
#/bin/rm -f ispc.aux ispc.log ispc.out ispc.tex
