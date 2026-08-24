#!/bin/bash

find . -type d -name venv -prune -o -type f -print0 | grep -z -i '\.identifier$' | xargs -0 rm -v