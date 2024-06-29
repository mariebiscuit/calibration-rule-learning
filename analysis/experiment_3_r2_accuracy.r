library(lme4)
library(tidyverse)
library(car)

setwd('/Users/alyssamarie/Desktop/school/research/calibration-rule-learning')
df <- read_csv("./results/experiment_3/r2_to_accuracy_data.csv")
df$is_llm <- as.logical(df$is_llm)
m <- lmer(r2~ accuracy*is_llm + (1|concept), data=df)
summary(m)
Anova(m)