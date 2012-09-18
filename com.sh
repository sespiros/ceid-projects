#!/bin/sh

sed -e:t -e's|\(.*\)/\*.*\*/|\1|;tt;/\/\*/!b;N;bt' savvas.orig > savvas;
