# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
# -W makes a warning an error and --keep-going still reports every one of them.
# The build is run this way locally (docs/build_docs.sh) and in CI, so they
# cannot differ. Add -n (nitpicky: every unresolved cross-reference is a
# warning) with SPHINXOPTS="-W --keep-going -n".
SPHINXOPTS    ?= -W --keep-going
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

copy_examples:
	mkdir -p ./source/examples
	rsync -a --delete ../examples/ ./source/examples/


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile copy_examples
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
