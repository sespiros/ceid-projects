#!/bin/bash
python tree.py
dot -Tpng tree.dot -o tree.png
gwenview tree.png
