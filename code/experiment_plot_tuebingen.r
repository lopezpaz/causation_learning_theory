load('tuebingen.data')

library(tikzDevice)

total <- sum(m[,6])

find_critical_value <- function(n) {
	i <- n
	while(binom.test(i, n, 0.5)$p.value < 0.05) 
		i <- i - 1
	i 
}

s <- 1 : total

criticals <- rep(0, total)

for (i in 1 : total)
	criticals[ i ] <- find_critical_value( i )

tikz("plot.tex", width=5,height=3, standAlone=TRUE)
par(mar=c(4,4,1,1),oma=rep(0,4))

plot((1 : total) / total* 100, criticals / s * 100, type = "l", lwd = 0, col =
"gray", xlim=c(0,100), ylim = c(0, 100), xlab="decission rate",
ylab="classification accuracy")

polygon(c((1 : total) / total* 100, rev((1 : total) / total * 100)),
c(criticals / s * 100, rep(50, total)), col = "gray", border = "gray")
polygon(c((1 : total) / total* 100, rev((1 : total) / total * 100)), c(100 -
criticals / s * 100, rep(50, total)), col = "gray", border = "gray")

lines(d_igci, col="red",  lwd=3)
lines(d_anm,  col="blue", lwd=3)
lines(d_rcc,  col="black",lwd=3)
lines(c(-50,150),c(50,50),lty=3)

legend("bottomright", c("RCC","ANM","IGCI"), lty=rep(1,3), lwd=c(3,3,3), col=c(1,"blue","red"))

dev.off()

system("pdflatex plot.tex")
