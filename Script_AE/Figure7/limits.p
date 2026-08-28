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
set ytics nomirror out autofreq offset 0.5,0,0
set border lw 2
set yrange [0:*]

# Start the first plot
set output "IOPSLimit.pdf"

set multiplot

set xlabel "ThreadNum" offset 0,1,0
set ylabel "Query per Second" offset 1,0,0

set y2tics out autofreq format '%gk'
set y2label "IOPS"
set y2range [0:700]
set key reverse Left
set boxwidth 0.5

set size 1.045, 0.92
set origin -0.01, 0.08

# set title "IOPS Limit" offset 0, -0.7
plot "IOPS_limit.csv" using 1:2:xtic(4) every ::1 with boxes title 'QPS' at 0.2, 0.07, \
     "IOPS_limit.csv" using 1:($3/1000.):xtic(4) every ::1 with linespoints title 'IOPS' at 0.8, 0.07 axes x1y2 ls 1 pointsize 0.5

unset multiplot