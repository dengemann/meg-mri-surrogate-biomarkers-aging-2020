library(plot3D) # This library will allow us to draw 3d plot
library(MASS)

source("./config.r")

Sigma <- matrix(c(10, 3, 3, 2), 2, 2)
n_samples <- 200
seed <- 5341223
X <- mvrnorm(n = n_samples, mu = rep(0, 2), Sigma = Sigma)
eps <- 0.4
mean_age <- 45
std_age <- 20
beta_mri <- c(-0.2, -0.23)
age_mri <- ((X %*% beta_mri + rnorm(n_samples) * eps) * std_age) + mean_age

beta_fmri <- c(0.2, -0.12)
age_fmri <- ((X %*% beta_fmri + rnorm(n_samples) * eps) * std_age) + mean_age

beta_meg <- c(0.24, .2)
age_meg <- ((X %*% beta_meg + rnorm(n_samples) * eps) * std_age) + mean_age
data <- data.frame(a = X[,1], b = X[,2], age = age_mri)

# Compute the linear regression (z = ax + by + d
plot_plane <- function(data, color){
  fit <- lm(age ~ a + b, data = data)

  grid.lines <- 25
  x.pred <- seq(min(data[, 'a']), max(data[, 'a']), length.out = grid.lines)
  y.pred <- seq(min(data[, 'b']), max(data[, 'b']), length.out = grid.lines)
  xy <- expand.grid(a = x.pred, b = y.pred)
  print(names(data))

  z.pred <- matrix(predict(fit, newdata = xy),
                   nrow = grid.lines, ncol = grid.lines)
  # fitted points for droplines to surface
  fitpoints <- predict(fit, data = data)
  # scatter plot with regression plane  
  scatter3D(
    data$a, data$b, data$age,
    pch = 18,
    cex = 2,
    col = color,
    # theta = 20, phi = 20,
    alpha = 0.5,
    ticktype = "detailed",
    # xlab = 'MRI1',
    # ylab = 'MRI2',
    zlab = "age",
    bty = 'n',
    surf = list(
        x = x.pred, y = y.pred, z = z.pred,
        facets = NA, fit = fitpoints),
    main = "")
}

data_mri <- data.frame(a = X[, 1], b = X[, 2], age = age_mri)
data_fmri <- data.frame(a = X[, 1], b = X[, 2], age = age_fmri)
data_meg <- data.frame(a = X[, 1], b = X[, 2], age = age_meg)

pdf("./figures/elements_fig_concept_scatter_meg.pdf")
plot_plane(data = data_meg, color = color_cats[['vermillon']])
dev.off()

pdf("./figures/elements_fig_concept_scatter_mri.pdf")
plot_plane(data = data_mri, color = color_cats[['blue']])
dev.off()

pdf("./figures/elements_fig_concept_scatter_fmri.pdf")
plot_plane(data = data_fmri, color = color_cats[['violet']])
dev.off()


fake_fmri <- data.frame(matrix(rnorm(50 * 10, 0, 1), 10, 20))
fmri_corr <- cor(fake_fmri)
diag(fmri_corr) <- NA
fmri_corr[lower.tri(fmri_corr)] <- NA

library(ggplot2)
library(reshape2)
plot_data <- melt(fmri_corr, na.rm = T)

ggplot(data = plot_data, mapping = aes(x = Var1, y = Var2,
       alpha = value)) +
  geom_tile(color= 'white', fill = color_cats['violet'],
            show.legend = F) +
  # scale_fill_viridis_c(option = 'plasma') + 
  theme_void()
  # scale_fill_distiller("RdPu")
ggsave("./figures/elements_fig_concept_connectivity_fmri.pdf",
       width = 5, height = 5)

psd_data <- read.csv("./viz_outputs/demo_meg_psd.csv")
psd_data$channel <- factor(psd_data$channel)

set.seed(42)
ggplot(
  # data = psd_data,
  data = subset(psd_data, channel %in% sample(unique(psd_data$channel), 40)) ,
  mapping = aes(x = freqs, y = psd, group = channel)) +
    geom_line(color = color_cats[['vermillon']], alpha = 0.1) +
    # scale_x_log10() +
    # xlim(2, 50) +
    theme_void()
ggsave("./figures/elements_fig_concept_power_meg.pdf",
       width = 5, height = 5)

#+ session_info
print(sessionInfo())