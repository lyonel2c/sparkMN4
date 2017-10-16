reset

#set title "small vs. big batch size"
set ylabel "speedup"
set xlabel "#nodes"

#set xtics nomirror
#set xtics 128
#set ytics 128
#set term png

#set term png truecolor enhanced font "Times,15"
#set term pngcairo dashed
set terminal pngcairo
set output "results.png"

set style data linespoints
#set sample 500
set yrange [1:128]

set logscale x 2
set logscale y 2
set grid
set termoption dashed

plot 	"linear.dat" using 1:2 pt 2 lc rgb "gray" title "Linear speedup" , \
		"alexnet.dat" using 1:2 dt 2 pt 4 lc rgb "black" title "AlexNet (B = 256)" , \
		"alexnet2.dat" using 1:2 pt 5 lc rgb "black" title "AlexNet (B = 1024)" , \
		"googlenet.dat" using 1:2 dt 2 pt 8 lc rgb "black" title "GoogLeNet (B = 256)", \
		"googlenet2.dat" using 1:2 pt 9 lc rgb "black" title "GoogLeNet (B = 1024)" 

