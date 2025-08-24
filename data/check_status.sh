#!/bin/bash

for i in * 
do 
    echo -e "------------------------------\n$i" 
    cat $i/note 
done
