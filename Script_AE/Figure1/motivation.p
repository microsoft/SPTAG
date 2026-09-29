# For a single column, set the width at 3.3 inches
# For across two columns, set the width at 7 inches

set terminal pdfcairo size 3.3, 1.75 font 'Linux Biolinum O,12'
# set terminal pdfcairo size 7, 1.75 font "UbuntuMono-Regular, 11"

# set default line style
set style line 1 lc rgb '#056bfa' lt 1 lw 1.7 pt 7 ps 1.5
set style line 2 lc rgb '#05a8fa' lt 1 lw 1.7 pt 7 ps 1.5
set style line 3 lc rgb '#fb8500' lt 1 lw 1.7 pt 7 ps 1.5
set style line 4 lc rgb '#ffb703' lt 1 lw 1.7 pt 7 ps 1.5
set style line 5 lc rgb '#b30018' lt 1 lw 1.7 pt 7 ps 1.5
set style line 6 lc rgb '#fa3605' lt 1 lw 1.7 pt 7 ps 1.5
set style line 7 lc rgb '#80d653' lt 1 lw 1.7 pt 7 ps 1.5
set style line 8 lc rgb '#9400d3' lt 1 lw 1.7 pt 7 ps 1.5
# set style line 1 lc rgb '#00d5ff' lt 1 lw 1.5 pt 7 ps 1.5
# set style line 2 lc rgb '#000080' lt 1 lw 1.5 pt 7 ps 1.5
# set style line 3 lc rgb '#ff7f0e' lt 1 lw 1.5 pt 7 ps 1.5
# set style line 4 lc rgb '#008176' lt 1 lw 1.5 pt 7 ps 1.5
# set style line 5 lc rgb '#b3b3b3' lt 1 lw 1.5 pt 7 ps 1.5
# set style line 6 lc rgb '#000000' lt 1 lw 1.5 pt 7 ps 1.5

# set grid style
set style line 20 lc rgb '#dddddd' lt 1 lw 1

set datafile separator ","
set encoding utf8
set autoscale
set grid ls 20
# set key box ls 20 opaque fc rgb "#3fffffff" width 2
set tics scale 0.5
set xtics nomirror out autofreq offset 0,0.5,0
set ytics nomirror out autofreq offset 0.5,0,0
set border 3 lw 2
# set xrange [0:*]
set yrange [0:*]

# Start the first plot
set output "Motivation.pdf"

set multiplot

set size 0.55, 0.88
set origin -0.03, 0.12

set xlabel "Latency (ms)" offset 0,1,0
set ylabel 'Recall 10\@10' offset 2,0,0
set xrange [1:5]
set xtics ("1" 1, "" 1.5, "2" 2, "" 2.5, "3" 3, "" 3.5, "4" 4, "" 4.5, "5" 5)
# set key bottom right reverse Left
unset key

# set title "Search Latency" offset 0, -0.7
unset title
set yrange [0.92:1]
set ytics 0.02
plot "motivation1.csv" using 3:4 every ::3 with linespoints title 'In-place update' ls 4 pointsize 0.5, \
     "motivation1.csv" using 1:2 every ::3 with linespoints title 'Static' ls 7 pointsize 0.5
unset key

set xlabel "Latency (ms)" offset 0,1,0
set ylabel "CDF" offset 2,0,0
set key bottom right reverse Left invert
set xrange [0:30]
set xtics 5
set yrange [0:*]
set ytics 0.2

set key
set size 0.56, 0.88
set origin 0.46, 0.12
# set title "Latency Distrubution After Data Shifting" offset 0, -0.7
unset title
plot "motivation.csv" using 1:2 every ::1 with linespoints title 'Static' at 0.25, 0.08 ls 7 pointsize 0.5, \
     "motivation.csv" using 3:4 every ::1 with linespoints title 'In-place update' at 0.65, 0.08 ls 4 pointsize 0.5

unset key

unset multiplot
