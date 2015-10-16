#!/bin/sh

sed -e:t -e's|\(.*\)/\*.*\*/|\1|;tt;/\/\*/!b;N;bt' test.buz > test.buz;
