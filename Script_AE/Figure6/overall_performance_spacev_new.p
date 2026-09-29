# For a single column, set the width at 3.3 inches
# For across two columns, set the width at 7 inches

# set terminal pdfcairo size 3.3, 2.07 font "UbuntuMono-Regular, 10"
set terminal pdfcairo size 7, 1.75 font 'Linux Biolinum O,12'

# set default line style
set style line 1 lc rgb '#056bfa' lt 1 lw 1.7 pt 7 ps 1.5
set style line 2 lc rgb '#05a8fa' lt 1 lw 1.7 pt 7 ps 1.5
set style line 3 lc rgb '#fb8500' lt 1 lw 1.7 pt 7 ps 1.5
set style line 4 lc rgb '#ffb703' lt 1 lw 1.7 pt 7 ps 1.5
set style line 5 lc rgb '#b30018' lt 1 lw 1.7 pt 7 ps 1.5
set style line 6 lc rgb '#fa3605' lt 1 lw 1.7 pt 7 ps 1.5
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
# set key box ls 20 opaque fc rgb "#3fffffff"
set tics scale 0.5
set xtics nomirror out autofreq offset 0,0.5,0
set xtics ("0" 0, "20" 20, "40" 40, "60" 60, "80" 80, "100" 100)
set xrange [0:100]
set ytics nomirror out autofreq offset 0.5,0,0
set border lw 2
set yrange [0:*]

# Start the first plot
set output "OverallPerformanceSPACEV1.pdf"
set multiplot

set xlabel "Batches/Days" offset 0,1,0
set ylabel "Latency (ms)" offset 1.1,0,0
# set key reverse Left
unset key

set size 0.26, 0.95
set origin 0, 0.13
set yrange [0:20]
set ytics ("0" 0, "" 2, "4" 4, "" 6, "8" 8, "" 10, "12" 12, "" 14, "16" 16, "" 18, "20" 20)
set title "Search Tail Latency" offset 0, -0.7
plot "overall_performance_spacev.csv" using 1:4 every ::3 with lines ls 1, \
     "overall_performance_spacev.csv" using 1:12 every ::3 with lines ls 3, \
     "overall_performance_spacev.csv" using 1:20 every ::3 with lines ls 5

set size 0.26, 0.95
set origin 0.25, 0.13
set ylabel 'Throughput (QPS)' offset 1.1,0,0 # {/Symbol m}
set yrange [0:1000]
set ytics 200
set title "Insert Throughput (per Thread)" offset 0, -0.7
plot "overall_performance_spacev2.csv" using 1:(1000000./$2) every ::3 with lines ls 1, \
     "overall_performance_spacev2.csv" using 1:(1000000./$5) every ::3 with lines ls 3, \
     "overall_performance_spacev2.csv" using 1:(1000000./$8) every ::3 with lines ls 5

# "overall_performance_spacev2.csv" using 1:($3/1000.) every ::3 with lines ls 2, \
# "overall_performance_spacev2.csv" using 1:($6/1000.) every ::3 with lines ls 4, \
# "overall_performance_spacev2.csv" using 1:($9/1000.) every ::3 with lines ls 6

set size 0.26, 0.95
set origin 0.49, 0.13
set ylabel 'Recall 10\@10' offset 2,0,0
set yrange [0.5:1]
set ytics 0.5,0.1,1
set title "Accuracy" offset 0, -0.7
plot "overall_performance_spacev2.csv" using 1:4 every ::3 with lines ls 1, \
     "overall_performance_spacev2.csv" using 1:7 every ::3 with lines ls 3, \
     "overall_performance_spacev2.csv" using 1:10 every ::3 with lines ls 5

set size 0.26, 0.95
set origin 0.73, 0.13
set ylabel "Memory (GB)" offset 2,0,0
set yrange [0:*]
set ytics autofreq
set key reverse Left
set title "Memory Usage" offset 0, -0.7
plot "overall_performance_spacev.csv" using 1:6 every ::3 with lines title 'SPFresh' at 0.35, 0.08 ls 1, \
     "overall_performance_spacev.csv" using 1:14 every ::3 with lines title 'DiskANN' at 0.5, 0.08 ls 3, \
     "overall_performance_spacev.csv" using 1:22 every ::3 with lines title 'SPANN+' at 0.65, 0.08 ls 5

unset key

unset multiplot
