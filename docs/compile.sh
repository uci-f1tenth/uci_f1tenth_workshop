#!/usr/bin/env bash
#Run the Script from the folder you are in...
CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DOC_DIR="$CURRENT_DIR/Development_Guide"
CMD_LATEX=lualatex
# avoid $TERM warning
export TERM=xterm-256color

# Change to the document directory so LaTeX finds mmp.sty
cd "$DOC_DIR" || {
	echo "Directory $DOC_DIR not found"
	exit 1
}

# Function to remove temporary files from previously crashed runs
cleanup() {
	rm -f ./*.bbl ./*.blg ./*.aux ./*.bcf ./*.ilg ./*.lof ./*.log ./*.lot ./*.nlo ./*.nls* ./*.out ./*.toc ./*.run.xml ./*.sub ./*.suc ./*.syc ./*.sym
}

# Remove existing PDF and temporary files at the beginning
rm -f development_guide.pdf
cleanup

echo "Compiling in Language: $1"
if [ "$1" = "en" ] || [ "$2" = "en" ]; then
	compile="$CMD_LATEX --shell-escape --jobname=development_guide '\\def\\FOMEN{}\\input{dissertation.tex}'"
	biberarg="development_guide"
else
	compile="$CMD_LATEX --shell-escape --jobname=development_guide '\\input{dissertation.tex}'"
	biberarg="development_guide"
fi

echo "Running: $compile"
eval "$compile"
RETVAL="$?"
if [[ "${RETVAL}" -ne 0 ]]; then
	echo "First $CMD_LATEX run failed"
	exit ${RETVAL}
fi

eval "$compile"
RETVAL="$?"
if [[ "${RETVAL}" -ne 0 ]]; then
	echo "Second $CMD_LATEX run failed"
	exit ${RETVAL}
fi

eval "$compile"
RETVAL="$?"
if [[ "${RETVAL}" -ne 0 ]]; then
	echo "Third $CMD_LATEX run failed"
	exit ${RETVAL}
fi

# Remove temporary files at the end
cleanup

mv development_guide.pdf "$CURRENT_DIR/Development_Guide.pdf"
echo "Moved PDF to $CURRENT_DIR/Development_Guide.pdf"
echo "Directory listing of $CURRENT_DIR:"
ls -l "$CURRENT_DIR"

echo "PDF Compile: Success"

exit 0
