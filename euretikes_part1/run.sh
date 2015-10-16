#!/bin/bash
python water.py
circo -Tpng graph.dot -o graph.png
gwenview graph.png
