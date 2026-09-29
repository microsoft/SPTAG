# For a single column, set the width at 3.3 inches
# For across two columns, set the width at 7 inches

set terminal pdfcairo size 3.3, 2.8 font 'Linux Biolinum O,12'
# set terminal pdfcairo size 7, 1.5 font "UbuntuMono-Regular, 11"

# set default line style
set style line 1 lc rgb '#056bfa' lt 1 lw 1.7 pt 7 ps 1.5
set style line 2 lc rgb '#05a8fa' lt 1 lw 1.7 pt 7 ps 1.5
set style line 3 lc rgb '#fb8500' lt 1 lw 1.7 pt 7 ps 1.5
set style line 4 lc rgb '#ffb703' lt 1 lw 1.7 pt 7 ps 1.5
set style line 5 lc rgb '#b30018' lt 1 lw 1.7 pt 7 ps 1.5
set style line 6 lc rgb '#fa3605' lt 1 lw 1.7 pt 7 ps 1.5

# set grid style
set style line 20 lc rgb '#dddddd' lt 1 lw 1

set datafile separator ","
set encoding utf8
set autoscale
set grid ls 20
# set key box ls 20 opaque fc rgb "#3fffffff" width 0.5
set tics scale 0.5
set xtics nomirror out autofreq offset 0,0.5,0 format ""
set ytics nomirror out autofreq offset 0.5,0,0
set border lw 2
# set yrange [0:*]

# Start the first plot
set output "StressTest.pdf"
set multiplot # layout 6, 1
set lmargin 10

unset xlabel
set yrange [3:6]
set ytics 1
# set ylabel "latency (ms)" offset 1.5,0,0
unset ylabel
set label 1 "Latency (ms)" at screen 0.05, graph 0.5 center rotate by 90
set key at graph 0.99, 0.37 reverse Left

unset key
set title "Search Tail Latency" offset 0, -0.7
set origin 0, 0.68
set size 1, 0.38
plot "stress_test.csv" using 1:4 every ::3 with lines ls 1

# Start the second plot

set yrange [0:4000]
set ytics ("0" 0, "1k" 1000, "2k" 2000, "3k" 3000, "4k" 4000)
# set ylabel "Query per Second" offset 1.5,0,0
set label 1 "Query per Second" at screen 0.05, graph 0.5 center rotate by 90
set title "Throughput" offset 0, -0.7
set origin 0, 0.3
set size 1, 0.48
set key at graph 0.99, 0.37 reverse Left
plot "stress_test.csv" using 1:5 every ::3 with lines title 'Search Throughput' ls 1, \
     "stress_test2.csv" using 1:(4*1000000./$2) every ::3 with lines title 'Insert Throughput' ls 2

# Start the third plot

set yrange [300:600]
set xlabel "Batches/Days" offset 0,1,0
# set ylabel 'IOPS' offset 1.5,0,0
set label 1 "IOPS" at screen 0.05, graph 0.63 center rotate by 90
set xtics format '%g'
set ytics out format '%gk'
set ytics ("400k" 400, "600k" 600)
set title "IOPS" offset 0, -0.7
# set key box ls 20 opaque fc rgb "#3fffffff" width 0.5
set key at graph 0.99, 0.35 reverse Left
set origin 0, 0
set size 1, 0.4
plot "stress_test.csv" using 1:11 every ::3 with lines notitle ls 1, \
     (400) with lines title 'Guaranteed Limit' ls 5 dashtype 2

unset key

unset multiplot