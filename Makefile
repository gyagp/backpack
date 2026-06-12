# Backpack build entry points.
#
# The Python build script owns provisioning and build directory placement.
# Generated build artifacts must stay under gitignore/.

PYTHON ?= python

.PHONY: all setup deps clone dawn runtime verify clean dawn-clean info help

all: runtime

setup:
	$(PYTHON) build.py setup

deps:
	$(PYTHON) build.py deps

clone:
	$(PYTHON) build.py clone

dawn:
	$(PYTHON) build.py dawn

runtime:
	$(PYTHON) build.py runtime

verify:
	$(PYTHON) build.py verify

clean:
	$(PYTHON) build.py clean

dawn-clean:
	$(PYTHON) build.py dawn-clean

info:
	$(PYTHON) build.py info

help:
	$(PYTHON) build.py help
