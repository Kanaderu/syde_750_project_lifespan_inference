#!/bin/sh

# Clean all tex files using latexmk
FOLDERS=`find -name "latexmkrc"`
for FOLDER in $FOLDERS; do
	pushd `dirname $FOLDER`
	for TEX in *.tex; do
		latexmk -C `basename $TEX .tex`
	done
	popd
done

find \( -name "*.run.xml" \
	-o -name "*.pyc" \
	-o -name "*.bbl" \
	-o -name "*.blg" \
	-o -name "*.bcf" \
	-o -name "*.aux" \
	-o -name "*.snm" \
	-o -name "*.out" \
	-o -name "*.toc" \
	-o -name "*.nav" \
	-o -name "*.vrb" \
	-o -name "*.fls" \
	-o -name "*.fdb_latexmk" \
	-o -name "*.log" \
	-o -name "*.backup" \) \
	-print \
	-delete
