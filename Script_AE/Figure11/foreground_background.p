# For a single column, set the width at 3.3 inches
# For across two columns, set the width at 7 inches

set terminal pdfcairo size 3.3, 1.75 font 'Linux Biolinum O,12'
# set terminal pdfcairo size 7, 2.07 font "UbuntuMono-Regular, 11"

# set default line style
set style line 1 lc rgb '#056bfa' lt 1 lw 1.7 pt 7 ps 1.5
set style line 2 lc rgb '#05a8fa' lt 1 lw 1.7 pt 7 ps 1.5
set style line 3 lc rgb '#fb8500' lt 1 lw 1.7 pt 7 ps 1.5
set style line 4 lc rgb '#ffb703' lt 1 lw 1.7 pt 7 ps 1.5
set style line 5 lc rgb '#b30018' lt 1 lw 1.7 pt 7 ps 1.5
set style line 6 lc rgb '#fa3605' lt 1 lw 1.7 pt 7 ps 1.5

# set grid style
set style line 20 lc rgb '#dddddd' lt 1 lw 1
set style fill solid

set datafile separator ","
set encoding utf8
set autoscale
set grid ls 20 noxtics ytics
# set key box ls 20 opaque fc rgb "#3fffffff"
set tics scale 0.5
set xtics nomirror out autofreq offset 0,0.5,0
set ytics nomirror out offset 0.5,0,0
set border lw 2
set yrange [0:6000]
set ytics 1500

set style data histogram
set style histogram cluster gap 1
set style fill solid
set boxwidth 0.9

# Start the first plot
set output "Scalability.pdf"

set multiplot

set xlabel "Foreground Update\nThread Num" offset 0,1,0
set ylabel "Throughput" offset 1,0,0

set key reverse Left

set size 0.6, 0.88
set origin -0.02, 0.12

# set title "Scalability (Background Thread = 1)" offset 0, -0.7
plot "foreground_background.csv" using 3:xtic(2) every ::1 title 'Foreground' at 0.2, 0.07, \
     "foreground_background.csv" using 4 every ::1 title 'Background' at 0.75, 0.07

set size 0.5, 0.88
set origin 0.52, 0.12
set xlabel "Background Update\nThread Num" offset 0,1,0
unset ylabel
set ytics format ""

# set title "Scalability (Background Thread = 8)" offset 0, -0.7
plot "foreground_background.csv" using 10:xtic(9) every ::1::3 title 'Foreground' at 0.2, 0.07, \
     "foreground_background.csv" using 11 every ::1::3 title 'Background' at 0.75, 0.07

unset multiplot