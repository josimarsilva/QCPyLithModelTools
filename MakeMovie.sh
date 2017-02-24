#!/bin/bash

#mkdir jpeg 
#echo *.eps | xargs -n1 pstopdf

cd Movies

ls Fault_CFF_All_TimeSteps_SlipRate_and_Geometry_1* | awk 'BEGIN {FS="Geometry_"} ; {print $2}' | awk 'BEGIN {FS="."} ; {print $1}' > filenumber.dat
ls Fault_CFF_All_TimeSteps_SlipRate_and_Geometry_2* | awk 'BEGIN {FS="Geometry_"} ; {print $2}' | awk 'BEGIN {FS="."} ; {print $1}' >> filenumber.dat
ls Fault_CFF_All_TimeSteps_SlipRate_and_Geometry_3* | awk 'BEGIN {FS="Geometry_"} ; {print $2}' | awk 'BEGIN {FS="."} ; {print $1}' >> filenumber.dat
ls Fault_CFF_All_TimeSteps_SlipRate_and_Geometry_4* | awk 'BEGIN {FS="Geometry_"} ; {print $2}' | awk 'BEGIN {FS="."} ; {print $1}' >> filenumber.dat
ls Fault_CFF_All_TimeSteps_SlipRate_and_Geometry_5* | awk 'BEGIN {FS="Geometry_"} ; {print $2}' | awk 'BEGIN {FS="."} ; {print $1}' >> filenumber.dat
ls Fault_CFF_All_TimeSteps_SlipRate_and_Geometry_6* | awk 'BEGIN {FS="Geometry_"} ; {print $2}' | awk 'BEGIN {FS="."} ; {print $1}' >> filenumber.dat
ls Fault_CFF_All_TimeSteps_SlipRate_and_Geometry_7* | awk 'BEGIN {FS="Geometry_"} ; {print $2}' | awk 'BEGIN {FS="."} ; {print $1}' >> filenumber.dat
ls Fault_CFF_All_TimeSteps_SlipRate_and_Geometry_8* | awk 'BEGIN {FS="Geometry_"} ; {print $2}' | awk 'BEGIN {FS="."} ; {print $1}' >> filenumber.dat
ls Fault_CFF_All_TimeSteps_SlipRate_and_Geometry_9* | awk 'BEGIN {FS="Geometry_"} ; {print $2}' | awk 'BEGIN {FS="."} ; {print $1}' >> filenumber.dat


while read filenumber
do
	filenamePNG="Fault_CFF_All_TimeSteps_SlipRate_and_Geometry_"$filenumber".png"
	filenameJPG="Fault_CFF_All_TimeSteps_SlipRate_and_Geometry_"$filenumber".jpg"


	if [ ! -f $filenameJPG ]; then
		if [  -f $filenamePNG ]; then
			sips -s format jpeg $filenamePNG --out .
			#echo $filename
		fi
	fi
	
	if [ ! -f $filenameJPG ]; then
		if [  -f $filenamePNG ]; then
			sips -s format jpeg $filenamePNG --out .
			#echo $filename
		fi
	fi
done < filenumber.dat


ls Fault_CumulativeDisplacement_1* | awk 'BEGIN {FS="CumulativeDisplacement_"} ; {print $2}' | awk 'BEGIN {FS="."} ; {print $1}' > filenumber.dat
ls Fault_CumulativeDisplacement_2* | awk 'BEGIN {FS="CumulativeDisplacement_"} ; {print $2}' | awk 'BEGIN {FS="."} ; {print $1}' >> filenumber.dat
ls Fault_CumulativeDisplacement_3* | awk 'BEGIN {FS="CumulativeDisplacement_"} ; {print $2}' | awk 'BEGIN {FS="."} ; {print $1}' >> filenumber.dat
ls Fault_CumulativeDisplacement_4* | awk 'BEGIN {FS="CumulativeDisplacement_"} ; {print $2}' | awk 'BEGIN {FS="."} ; {print $1}' >> filenumber.dat
ls Fault_CumulativeDisplacement_5* | awk 'BEGIN {FS="CumulativeDisplacement_"} ; {print $2}' | awk 'BEGIN {FS="."} ; {print $1}' >> filenumber.dat
ls Fault_CumulativeDisplacement_6* | awk 'BEGIN {FS="CumulativeDisplacement_"} ; {print $2}' | awk 'BEGIN {FS="."} ; {print $1}' >> filenumber.dat
ls Fault_CumulativeDisplacement_7* | awk 'BEGIN {FS="CumulativeDisplacement_"} ; {print $2}' | awk 'BEGIN {FS="."} ; {print $1}' >> filenumber.dat
ls Fault_CumulativeDisplacement_8* | awk 'BEGIN {FS="CumulativeDisplacement_"} ; {print $2}' | awk 'BEGIN {FS="."} ; {print $1}' >> filenumber.dat
ls Fault_CumulativeDisplacement_9* | awk 'BEGIN {FS="CumulativeDisplacement_"} ; {print $2}' | awk 'BEGIN {FS="."} ; {print $1}' >> filenumber.dat


while read filenumber
do
        filenamePNG="Fault_CumulativeDisplacement_"$filenumber".png"
        filenameJPG="Fault_CumulativeDisplacement_"$filenumber".jpg"

        if [ ! -f $filenameJPG ]; then
                if [  -f $filenamePNG ]; then
                        sips -s format jpeg $filenamePNG --out .
                        #echo $filename
                fi
        fi

        if [ ! -f $filenameJPG ]; then
                if [  -f $filenamePNG ]; then
                        sips -s format jpeg $filenamePNG --out .
                        #echo $filename
                fi
        fi
done < filenumber.dat


ls Fault_CumulativeStressChange_1* | awk 'BEGIN {FS="CumulativeStressChange_"} ; {print $2}' | awk 'BEGIN {FS="."} ; {print $1}' > filenumber.dat
ls Fault_CumulativeStressChange_2* | awk 'BEGIN {FS="CumulativeStressChange_"} ; {print $2}' | awk 'BEGIN {FS="."} ; {print $1}' >> filenumber.dat
ls Fault_CumulativeStressChange_3* | awk 'BEGIN {FS="CumulativeStressChange_"} ; {print $2}' | awk 'BEGIN {FS="."} ; {print $1}' >> filenumber.dat
ls Fault_CumulativeStressChange_4* | awk 'BEGIN {FS="CumulativeStressChange_"} ; {print $2}' | awk 'BEGIN {FS="."} ; {print $1}' >> filenumber.dat
ls Fault_CumulativeStressChange_5* | awk 'BEGIN {FS="CumulativeStressChange_"} ; {print $2}' | awk 'BEGIN {FS="."} ; {print $1}' >> filenumber.dat
ls Fault_CumulativeStressChange_6* | awk 'BEGIN {FS="CumulativeStressChange_"} ; {print $2}' | awk 'BEGIN {FS="."} ; {print $1}' >> filenumber.dat
ls Fault_CumulativeStressChange_7* | awk 'BEGIN {FS="CumulativeStressChange_"} ; {print $2}' | awk 'BEGIN {FS="."} ; {print $1}' >> filenumber.dat
ls Fault_CumulativeStressChange_8* | awk 'BEGIN {FS="CumulativeStressChange_"} ; {print $2}' | awk 'BEGIN {FS="."} ; {print $1}' >> filenumber.dat
ls Fault_CumulativeStressChange_9* | awk 'BEGIN {FS="CumulativeStressChange_"} ; {print $2}' | awk 'BEGIN {FS="."} ; {print $1}' >> filenumber.dat


while read filenumber
do
        filenamePNG="Fault_CumulativeStressChange_"$filenumber".png"
        filenameJPG="Fault_CumulativeStressChange_"$filenumber".jpg"

        if [ ! -f $filenameJPG ]; then
                if [  -f $filenamePNG ]; then
                        sips -s format jpeg $filenamePNG --out .
                        #echo $filename
                fi
        fi

        if [ ! -f $filenameJPG ]; then
                if [  -f $filenamePNG ]; then
                        sips -s format jpeg $filenamePNG --out .
                        #echo $filename
                fi
        fi
done < filenumber.dat





ffmpeg -r 3  -i Fault_CFF_All_TimeSteps_SlipRate_and_Geometry_%d.jpg Movie_FaultStress.mp4
ffmpeg -r 3  -i Fault_CumulativeDisplacement_%d.jpg Movie_CummulativeDisplacement.mp4
ffmpeg -r 3  -i Fault_CumulativeStressChange_%d.jpg Movie_CummulativeStressChange.mp4


#sips -s format jpeg *.png --out .
#ffmpeg -r 3  -i Stress_Fig_%d.jpg Movie.mp4

cd ..

