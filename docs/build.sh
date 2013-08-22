#!/bin/bash

rst2html=rst2html.py

for i in ispc perfguide faq; do
    $rst2html --template=template.txt --link-stylesheet \
        --stylesheet-path=css/style.css $i.rst > $i.html
done

$rst2html --template=template-news.txt --link-stylesheet \
    --stylesheet-path=css/style.css news.rst > news.html

$rst2html --template=template-perf.txt --link-stylesheet \
        --stylesheet-path=css/style.css perf.rst > perf.html

#rst2latex --section-numbering --documentclass=article --documentoptions=DIV=9,10pt,letterpaper ispc.txt > ispc.tex
#pdflatex ispc.tex
#/bin/rm -f ispc.aux ispc.log ispc.out ispc.tex
