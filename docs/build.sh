#!/bin/bash

rst2html=rst2html

for i in ispc ispc_for_gen perfguide faq; do
    $rst2html --template=template.txt --link-stylesheet \
        --stylesheet-path=css/style.css $i.rst > $i.html
done

$rst2html --template=template-perf.txt --link-stylesheet \
        --stylesheet-path=css/style.css perf.rst > perf.html

#rst2latex --section-numbering --documentclass=article --documentoptions=DIV=9,10pt,letterpaper ispc.rst > ispc.tex
#pdflatex ispc.tex
#/bin/rm -f ispc.aux ispc.log ispc.out ispc.tex
