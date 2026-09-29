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
set key box ls 20 opaque width -2
set tics scale 0.5
set xtics nomirror out autofreq offset 0,0.5,0
set ytics nomirror out autofreq offset 0.5,0,0
set border lw 2

# Start the first plot
set output "ParameterStudy1.pdf"

set xlabel "Latency (ms)" offset 0,1,0
set ylabel 'Recall 10\@10' offset 1.5,0,0
set key bottom right reverse Left

set size 1, 1
set origin 0, 0
# set title "Search Latency" offset 0, -0.7
unset title
set yrange [0.92:1]
set ytics 0.02
plot "parameter_study_shifting.csv" using 3:4 every ::3 with lines title 'Static' ls 7, \
     "parameter_study_shifting.csv" using 1:2 every ::3 with lines title 'In-place Update' ls 4, \
     "parameter_study_shifting.csv" using 5:6 every ::3 with lines title 'In-place Update + Split Only' ls 3, \
     "parameter_study_shifting.csv" using 7:8 every ::3 with lines title 'In-place Update + Split/Reassign' ls 1

unset key