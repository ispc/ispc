#!/bin/bash

rst2html=rst2html

exit_status=0
for i in ispc ispc_for_xe perfguide faq design/invoke_sycl; do
    $rst2html --template=template.txt --link-stylesheet \
        --stylesheet-path=css/style.css --exit-status=2 $i.rst > $i.html
    (( exit_status = exit_status || $? ))
done

$rst2html --template=template-perf.txt --link-stylesheet \
        --stylesheet-path=css/style.css --exit-status=2 perf.rst > perf.html
(( exit_status = exit_status || $? ))

#rst2latex --section-numbering --documentclass=article --documentoptions=DIV=9,10pt,letterpaper ispc.rst > ispc.tex
#pdflatex ispc.tex
#/bin/rm -f ispc.aux ispc.log ispc.out ispc.tex

[ $exit_status -eq 0 ]
