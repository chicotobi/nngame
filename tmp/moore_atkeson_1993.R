library(tidyverse)
library(magrittr)

opti <- read.table("optimum_path.tsv",header=T) %>% rename(xx=x,yy=y)

opti %>% ggplot(aes(xx,yy)) + geom_point() + coord_equal()

d2 <- opti %>% mutate(x_1=xx+3*cos(theta/180*pi),y_1=yy+3*sin(theta/180*pi),x_2=2*xx-x_1,y_2=2*yy-y_1) %>%
  mutate(idx=row_number()) %>% pivot_longer(c(x_1,x_2,y_1,y_2)) %>% separate(name,c("var","n")) %>%
  pivot_wider(c(idx,n),names_from=var,values_from=value)

obstacles <- read.csv("~/Downloads/obstacles.csv",header=F) %>% mutate(idx=ceiling(row_number()/4))
colnames(obstacles) <- c("x","y","idx")
obstacles %<>% mutate(x=round(x,2),y=round(y,2))

d2 %>% ggplot(aes(x,y,grp=factor(idx))) + geom_path() + coord_equal(xlim=c(0,19),ylim=c(0,19),expand = F) + geom_point(data=opti%>%mutate(idx=NA),aes(xx,yy),col="red") +
  geom_polygon(data=obstacles,mapping=aes(x=x,y=y)) +
  scale_x_continuous(breaks=seq(0,19),minor_breaks=NULL) + scale_y_continuous(breaks=seq(0,19),minor_breaks=NULL)
