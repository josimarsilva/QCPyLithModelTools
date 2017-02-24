#!/bin/bash

#mkdir jpeg 
#echo *.eps | xargs -n1 pstopdf

cd Frames

sips -s format jpeg *.png --out .

ffmpeg -r 3  -i Fault_CFF_All_TimeSteps_SlipRate_and_Geometry_%d.jpg Movie.mp4

#ffmpeg -r 3  -i Stress_Fig_%d.jpg Movie.mp4

cd ..

