#!/bin/bash

mkdir jpeg 
echo *.eps | xargs -n1 pstopdf
sips -s format jpeg *.pdf --out ./jpeg/

cd jpeg

ffmpeg -r 3  -i Stress_Fig_%d.jpg Movie.mp4

cd ..

