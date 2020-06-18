#'---
#'title: "2.1 MRI, fMRI and MEG"
#'author: "Denis A. Engemann"
#'date: "8/6/2019"
#'output:
#'    html_document:
#'        code_folding:
#'            hide
#'    md_document:
#'        variant:
#'            markdown_github
#'---

# seed!
set.seed(42)

#+ config
library(ggplot2)
library(ggbeeswarm)
library(ggrepel)

# imports `color_cats` and `get_label_from_marker`
source('./utils.r')
source('./config.r')

PREDICTIONS <- './data/age_stacked_predictions_megglobal.csv'
PREDICTIONS2 <- './data/age_stacked_predictions__na_coded.csv'
PREDICTIONS3 <- './data/age_stacked_predictions_meglocal.csv'

DUMMY <- './data/age_stacked_dummy.csv'
SCORES <- './data/age_stacked_scores_megglobal.csv'

data_dummy <- read.csv(DUMMY)
data_dummy$MAE <- -data_dummy$MAE
  
data_pred_wide <- read.csv(PREDICTIONS)
data_scores_wide <- read.csv(SCORES)

# for chance-level predictiong and differences between methods

# fix column names and add dummy

names(data_scores_wide) <- fix_column_names(names(data_scores_wide))
data_scores_wide$chance <- data_dummy$MAE

#'Preprocess data.

#'r data
stacked_keys <- c(
  "MEG_handcrafted",
  "MEG_powers",
  "MEG_powers_cross_powers",
  "MEG_powers_cross_powers_handrafted",
  "MEG_cat_powers_cross_powers_correlation",
  "MEG_cat_powers_cross_powers_correlation_handcrafted",
  "MEG_cross_powers_correlation",
  "MEG_powers_cross_powers_correlation",
  "MEG_all",
  "ALL",
  "ALL_no_fMRI",
  "MRI",
  "ALL_MRI",
  "fMRI",
  "chance"
)

#' Let's first first get an overview on all models

base_line_sel <- c(
 "ALL", "ALL_no_fMRI",  "ALL_MRI", "MRI", "fMRI",  "MEG_all", "chance")

data_all_scores <- data_scores_wide[,base_line_sel]
names(data_all_scores) <- c('Multimodal', 'MRI & MEG', 'MRI & MRI', "MRI", 'fMRI', "MEG", "Chance")

#' Now let's investigate difference from chance baseline

summarize_table <- function(data){
    data <- rbind(
      apply(data, FUN = mean, 2),
      apply(data, FUN = sd, 2),
      apply(data, FUN = function(x) quantile(x, c(0.025, 0.975)), 2)
    )

    rownames(data) <- c("M", "SD", rownames(data)[c(3, 4)])
    data <- round(t(data), 3)
    data <- data.frame(model = rownames(data), data)
    rownames(data) <- NULL
    return(data)
}

data_all_summary <- summarize_table(data_all_scores)

knitr:::kable(data_all_summary)

write.csv(
  data.frame(
  stat = rownames(data_all_summary), data_all_summary),
  './viz_intermediate_files/all_scores_summary.csv'
)

data_dummy_diff <- data_all_scores - data_all_scores$Chance

data_dummy_diff_summary <- summarize_table(data_dummy_diff)

data_dummy_diff_summary$Pr <- t(t(apply(data_dummy_diff, FUN = function(x) sum(x < 0), 2)))

knitr:::kable(data_dummy_diff_summary)

write.csv(
  data.frame(
  stat = rownames(data_dummy_diff_summary), data_dummy_diff_summary),
  './viz_intermediate_files/all_scores_dummy_diff_summary.csv'
)


data_mri_diff <- data_all_scores - data_all_scores$MRI

data_mri_diff_summary <- summarize_table(data_mri_diff)

data_mri_diff_summary$Pr <- t(t(apply(data_mri_diff, FUN = function(x) sum(x < 0), 2)))

knitr:::kable(data_mri_diff_summary)

write.csv(
  data.frame(
  stat = rownames(data_mri_diff_summary), data_mri_diff_summary),
  './viz_intermediate_files/all_scores_mri_diff_summary.csv'
)


# XXX idea, rank statistics
model_ranking <- do.call(
  rbind, 
  lapply(seq_along(data_all_scores[,1]),
         function(ii) {
          out <- data.frame(
            rank = rank(data_all_scores[ii,]),
            model = names(data_all_scores),
            fold = ii)
          rownames(out) <- NULL
          return(out)
        }))

model_ranking$model <- factor(
  model_ranking$model,
  levels = rev(c("Chance", "fMRI", "MEG", "MRI", "MRI & MEG",
                 "MRI & MRI", "Multimodal")),
  labels = rev(c(
    "Chance", "fMRI", "MEG", "MRI", "MRI, MEG", "MRI, fMRI",
    "MRI, fMRI, MEG")))

colors_multimodal <- setNames(
  with(color_cats, c(black, orange, `blueish green`, blue, violet, vermillon, gray)),
  c("MRI, fMRI, MEG", 'MRI, fMRI', 'MRI, MEG', 'MRI', 'fMRI', 'MEG', 'Chance'))

fig_rank_box <- ggplot(data = model_ranking,
       mapping = aes(x = rank, y = reorder(model, rank, mean), color = model, fill = model)) +
geom_boxplot(show.legend = F, alpha = 0.3) +
geom_jitter(show.legend = F, size=1.5, width = 0.12, alpha=0.5) +
scale_x_continuous(breaks = 1:7) +
scale_color_manual(values = colors_multimodal,
                   labels = names(colors_multimodal),
                   breaks = names(colors_multimodal)) +
scale_fill_manual(values = colors_multimodal,
                  labels = names(colors_multimodal),
                  breaks = names(colors_multimodal)) +
labs(x = 'Ranking across CV testing-splits', y = 'Stacking Models')
print(fig_rank_box)


fname <- "./figures/elements_fig2_stacking_mri_meg_supp_rank_box."
ggsave(paste0(fname, "pdf"),
       width = save_width, height = save_height, useDingbats = F)
embedFonts(file = paste0(fname, "pdf"), outfile = paste0(fname, "pdf"))
ggsave(paste0(fname, "png"),
       width = save_width, height = save_height, dpi = 300)
knitr::include_graphics(paste0(fname, "png"), dpi = 200)

##
# explore rank stats
rankmat <- data.frame(
  do.call(
    cbind,
    with(model_ranking,
      by(model_ranking, model, function(x) x$rank))))
rankmat$n <- 1
rank_ds <- destat(rankmat)
rownames(rank_ds$pair) <- names(rankmat)[-ncol(rankmat)]
colnames(rank_ds$pair) <- names(rankmat)[-ncol(rankmat)]
rownames(rank_ds$mar) <- names(rankmat)[-ncol(rankmat)]
colnames(rank_ds$mar) <- names(rankmat)[-ncol(rankmat)]

print(rank_ds$pair)
print(rank_ds$mar)

get_mat <- function(mat) {
  mat <- data.frame(mat)
  colnames(mat) <- levels(model_ranking$model)
  rownames(mat) <- levels(model_ranking$model)

  mat_long <- do.call(rbind, lapply(colnames(mat), function(x) {
    data.frame(count = mat[, x], x = x, y = rownames(mat))
  }))
  mat_long$x <- factor(mat_long$x, levels = colnames(mat))
  mat_long$y <- factor(mat_long$y, levels = colnames(mat))
  mat_long
}

fig_rank_mat_pair <- ggplot(data = get_mat(rank_ds$pair),
                            mapping = aes(x = x, y = y, fill = count, label = count)) +
  geom_tile() + 
  scale_fill_viridis_c(begin = 0.01, end = 0.99) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  geom_text(color = 'white', size = 8) +
  labs(x = element_blank(), y = element_blank())
print(fig_rank_mat_pair)

fname <- "./figures/elements_fig2_stacking_mri_meg_supp_rank_mat_pair."
ggsave(paste0(fname, "pdf"),
       width = save_width, height = save_height, useDingbats = F)
embedFonts(file = paste0(fname, "pdf"), outfile = paste0(fname, "pdf"))
ggsave(paste0(fname, "png"),
       width = save_width, height = save_height, dpi = 300)
knitr::include_graphics(paste0(fname, "png"), dpi = 200)


#' Now we can investigate the differences between models in depth.
reshape_sel <- names(data_scores_wide)[!names(data_scores_wide) %in% c(
  'repeat_', 'repeat_idx', 'age')]

data_scores <- reshape(data = data_scores_wide,
                       direction = 'long',
                       varying = reshape_sel,
                       v.names = 'MAE',
                       timevar = 'marker',
                       times = reshape_sel)

data_scores['modality'] <- 'MEG'
data_scores[grepl('MEG_', data_scores$marker),]['modality'] <- 'MEG'
data_scores[grepl('Connectivity_Matrix', 
                  data_scores$marker),]['modality'] <- 'fMRI'
data_scores[grepl('ALL', data_scores$marker),]['modality'] <- 'Multimodal'
data_scores[grepl('chance', data_scores$marker),]['modality'] <- 'Chance'

data_scores$marker <- sub("stacked_", "", data_scores$marker)
data_scores$prediction <- factor(ifelse(
  data_scores$marker %in% stacked_keys, 'stacked', 'linear'))
data_scores$marker <- factor(data_scores$marker)

#'Select stacked data.

#+r stacked_data
data_stacked <- subset(data_scores, prediction == 'stacked')

stacked_selection <- c(
  "ALL",
  "ALL_MRI",
  "ALL_no_fMRI",
  "MRI",
  "chance"
)

data_stacked_sel <- within(
    data_stacked[data_stacked$marker %in% stacked_selection,],
    {
      family <- rep('Multimodal', length(marker))
      family[marker == 'ALL_MRI'] <- 'MRI & fMRI'
      family[marker == 'ALL_no_fMRI'] <- 'MRI & MEG'
      family[marker == 'MRI'] <- 'MRI'
      family[marker == 'chance'] <- 'Chance'
      family <- factor(family)
    }
)

#'Plot error distibution.
#+ fig2a
colors <- setNames(
  with(color_cats, c(black, orange, `blueish green`, blue, gray)),
  c('Multimodal', 'MRI & fMRI', 'MRI & MEG', 'MRI', "Chance"))

sort_idx <- order(aggregate(MAE ~ marker,
                            data = data_stacked_sel, FUN = mean)$MAE)
data_stacked_sel$cv_idx <- rep(rep(1:10, times = 10), length(colors))


set_na_chance <- function(data){
  data$is_chance <- F

  data <- rbind(
    data,
    within(data, {
      is_chance <- T
      MAE[family !=  "Chance"] <- NA})
  )

  data[
    (data$is_chance == F &
     data$family == "Chance"),]$MAE <- NA
  return(data)
}

sort_idx <- order(
      aggregate(MAE~marker, data = data_stacked_sel, FUN = mean)$MAE)

data_stacked_sel$marker <- factor(
  data_stacked_sel$marker,
  levels(factor(data_stacked_sel$marker))[sort_idx])

data_stacked_sel <- subset(data_stacked_sel, family != "Chance")

mri_mean <- aggregate(MAE ~ family, data_stacked_sel, mean)[1, 2]


#'Now let us look at the difference from baseline.

#+ fig2b
# The power of R ... do multi-line assignment expressions inside data frame
# environment and return updated data.frame

data_diff <- within(subset(data_stacked_sel, family != "Chance"),
    {
      MAE_diff <- c(MAE[family == 'Multimodal'] - MAE[family == 'MRI'],
                    MAE[family == 'MRI & MEG'] - MAE[family == 'MRI'],
                    mri_mean - MAE[family == 'MRI'],
                    MAE[family == 'MRI & fMRI'] - MAE[family == 'MRI'])
      family <- factor(gsub("Multimodal", "Fully multimodal", family))
    }
)

legend_name <- "Improvement over anatomical MRI"
colors_fig2b <- setNames(
  with(color_cats,
       c(black, orange, `blueish green`, blue)),
  c("MRI, fMRI, MEG", 'MRI, fMRI', 'MRI, MEG', 'MRI'))

sort_idx <- order(aggregate(MAE_diff ~ family, data_diff, mean)$MAE_diff)

levels(data_diff$family) <- c("MRI, fMRI, MEG", "MRI", "MRI, fMRI", "MRI, MEG")


fig2b <- ggplot(
  data = data_diff,
  mapping = aes(y = MAE_diff,
                x = reorder(family, MAE_diff, function(x) mean(x)),
                color = family, fill = family)) +
  coord_flip(ylim = c(-2.9, 1.7)) +
  stat_summary(geom = "boxplot", fun.data = my_quantiles,
               alpha = 0.5, size = 0.7, width = 0.8) +
  stat_summary(geom = "errorbar", fun.data = my_quantiles,
               alpha = 0.5, size = 0.7, width = 0.5) +
  stat_summary(geom = 'text',
               mapping = aes(label  = sprintf("%1.1f",
                                              ..y.. +
                                              mri_mean)),
               fun.y= mean, size = 3.2, show.legend = FALSE,
               position = position_nudge(x=-0.49)) +
  geom_beeswarm(alpha=0.3, show.legend = F, size = 3) +
  geom_hline(yintercept = 0, linetype = 'dashed') +
  guides(color = guide_legend(nrow = 3, title.position = "top")) +
  ylab("MAE difference (years)") +
  xlab("Multimodal stacking") +
  scale_color_manual(values = colors_fig2b,
                     labels = names(colors_fig2b),
                     breaks = names(colors_fig2b),
                     name = legend_name) +
  scale_fill_manual(values = colors_fig2b,
                    labels = names(colors_fig2b),
                    breaks = names(colors_fig2b),
                    name = legend_name) +
  guides(color = guide_legend(nrow = 1, title.position = 'top'),
         fill = guide_legend(nrow = 1)) +
  scale_y_continuous(breaks = seq(-3, 1.5, 0.5)) +
  theme(axis.text.y = element_blank(),
        legend.position = 'top',
        legend.justification = 'left',
        legend.text.align = 0)
print(fig2b)

fname <- "./figures/elements_fig2_stacking_mri_meg_diff."
ggsave(paste0(fname,  "pdf"),
       width = save_width, height = save_height, useDingbats = F)
embedFonts(file = paste0(fname, "pdf"), outfile = paste0(fname, "pdf"))
ggsave(paste0(fname,  "png"), width = save_width, height = save_height,
       dpi = 300)
knitr::include_graphics(paste0(fname, "png"), dpi = 200)

#'Let's investigate the interaction between the MEG and MRI.
#'For this we will have to extract the raw prediction errors.
#'
#'We will first prepare a subset of data that allows us to
#'compare MEG and fRMI.

#+ data_wide
stacked_selection2 <- c(stacked_selection, "fMRI", "MEG_all")
data_pred_stacked_sel <- preprocess_prediction_data(
  df_wide = data_pred_wide, stack_sel = stacked_selection2, drop_na = T)

#'Now we can package the data for plotting.

#+ data_diff
data_pred_stacked_comp <- rbind(
  data.frame(
    MAE_meg = subset(data_pred_stacked_sel, family == 'MEG')$MAE,
    MAE_fmri = subset(data_pred_stacked_sel, family == 'fMRI')$MAE,
    combined = F),
   data.frame(
    MAE_meg = subset(data_pred_stacked_sel, family == 'MRI & MEG')$MAE,
    MAE_fmri = subset(data_pred_stacked_sel, family == 'MRI & fMRI')$MAE,
    combined = T)
)

data_pred_stacked_comp <- cbind(
  data_pred_stacked_comp,
  rbind(subset(data_pred_stacked_sel, family == 'MEG', select = -c(MAE)),
        subset(data_pred_stacked_sel, family == 'MEG', select = -c(MAE))))

data_pred_stacked_comp$combined <- factor(
  ifelse(data_pred_stacked_comp$combined, "MRI[anat.]~added", "no~MRI[anat.]"),
  levels = c("no~MRI[anat.]", "MRI[anat.]~added"))

#'Now we can plot it.
data_corr_agg <- aggregate(
    cbind(MAE_fmri, MAE_meg, age) ~ X*combined,
    data = data_pred_stacked_comp, FUN = mean)


cor1 <- with(subset(data_corr_agg, combined != "MRI[anat.]~added"),
             cor.test(MAE_meg, MAE_fmri,  method = "spearman"))
cor2 <- with(subset(data_corr_agg, combined == "MRI[anat.]~added"),
             cor.test(MAE_meg, MAE_fmri,  method = "spearman"))

cor.details <- data.frame(
  p.value = c(sprintf("%e", cor1$p.value),
              ifelse(cor2$p.value == 0, "2.2e-16", sprintf("%e", cor2$p.value))),
  r2 = round(c(cor1$estimate, cor2$estimate) ^ 2, 3),
  rho = round(c(cor1$estimate, cor2$estimate), 3),
  mri = c(F, T)
)

write.csv(cor.details, './viz_intermediate_files/correlation_meg_mri.csv')

# annotation <- data.frame(
#    x = c(2,4.5),
#    y = c(20,25),
#    label = c("label 1", "label 2")
# )

#+ fig2c
fig2c <- ggplot(
  data = data_corr_agg,
  mapping = aes(x = MAE_fmri, y =  MAE_meg, size = age, color = age)) +
  geom_point(alpha = 0.8) +
  scale_size_continuous(range = c(0.01, 3),
                        trans = 'sqrt') +
  ylab(expression(MAE[MEG] ~ (years))) +
  xlab(expression(MAE[fMRI] ~ (years))) +
  facet_wrap(~combined, labeller = label_parsed) +
  scale_color_viridis_c() +
  theme(legend.position = 'top',
        legend.justification = 'left',
        legend.text.align = 0) +
  guides(
        color = guide_legend(title.position = "left"),
        size = guide_legend(title.position = "left")) +
  coord_fixed(ylim = c(0, 30.5), xlim = c(0, 30.5)) +
  labs(color = "age", shape = "age")
print(fig2c)

fname <- "./figures/elements_fig2_supplement_mri_meg_scatter."
ggsave(paste0(fname, "pdf"), plot = fig2c,
       width = save_width, height = save_height, useDingbats = F)
embedFonts(file = paste0(fname, "pdf"), outfile = paste0(fname, "pdf"))
ggsave(paste0(fname, "png"), plot = fig2c,
        width = save_width, height = save_height,
       dpi = 300)
knitr::include_graphics(paste0(fname, "png"), dpi = 200)

#'One can see that a) the MEG and MRI errors are not
#'strongly related b) comabining each of them with MRI
#'Makes the error somewhat more similar, especially in old
#'people, yet leaves them rather  uncorrelated.
#'

#'Let us now plot the error by age group and modality.

#+ error_by_age
# make qualitative ordered age group

colors_fig2e <- setNames(
  with(color_cats, c(black, orange, `blueish green`, blue, violet, vermillon)),
  c("MRI, fMRI, MEG", 'MRI, fMRI', 'MRI, MEG', 'MRI', 'fMRI', 'MEG'))

data_pred_stacked_sel$age_group <- cut(data_pred_stacked_sel$age, breaks = seq(15, 90, 5))
data_pred_stacked_sel$family <- factor(
  data_pred_stacked_sel$family,
  levels = c("MRI", "fMRI", "MEG", "MRI & fMRI", "MRI & MEG", "Multimodal"),
  labels = c("MRI", "fMRI", "MEG", "MRI, fMRI", "MRI, MEG", "MRI, fMRI, MEG"))


names(data_pred_stacked_sel) <- c("subject", names(data_pred_stacked_sel)[-1])

data_pred_stacked_sel_agg <- aggregate(
    cbind(MAE, age) ~ subject*family, data = data_pred_stacked_sel, FUN = mean)

data_pred_stacked_sel_agg$age_group <- cut_number(data_pred_stacked_sel_agg$age, 7)

(anomod <- summary(aov(log(MAE) ~ age_group * family,
                  data = data_pred_stacked_sel_agg)))

writeLines(capture.output(anomod),
           "./viz_intermediate_files/anova_stacking_error.txt")


fig2d <- ggplot(data = data_pred_stacked_sel,
                mapping = aes(x = age, y = MAE
                              # size = MAE, 
                              # color = family, 
                              # fill = family
                            )) +
  facet_wrap(.~family, scales = 'free_x') +

  # geom_jitter(shape=21, fill='white', show.legend = F, alpha=0.5) +

  geom_hex(mapping =  aes(x = age, y = MAE),
           show.legend = T, size = 0.1, bins = 15) +
  stat_smooth(size = 1.2, show.legend = F,
              method = loess, 
              color='red',
              method.args = list(degree = 2),
              fill = NA, level = .9999) +
  scale_fill_continuous(type = "viridis") +
  theme(strip.text.y = element_text(angle = 0),
        panel.spacing.x = unit(0.05, 'in')) +
  xlab("Age (years)") +
  ylab("MAE (years)")
  # facet_wrap(~family, ncol = 3)
print(fig2d)

fname <- "./figures/elements_fig2_error_by_age_group."
ggsave(paste0(fname, "pdf"), plot = fig2d,
       width = save_width, height = save_height, useDingbats = F)
ggsave(paste0(fname, "png"), plot = fig2d,
        width = save_width, height = save_height,
       dpi = 300)
knitr::include_graphics(paste0(fname, "png"), dpi = 200)

#' We see, what we knew before that, stacking helps improve the error
#' However, we see that this is probably due to at least two mechanisms.
#' 1) extreme error is unifomely reduced. 2) Error in young and old groups
#' is mitigated. However, we also learn that the best model still shows
#' considerable brain age bias, with young and old subjects systematically
#' suffering from more error.

#' Time to investigate the 2D dependence. To select the right variables,

#' We can see from the printed outputs that MEG power always makes it under
#' the top markers. Thus, it makes sense to focus on power in a 2D dependence
#' analysis.

OUT_DEPENDENCE_2D <- './data/age_stacked_dependence_model-full-2d.csv'
data_dependence2d <- read.csv(OUT_DEPENDENCE_2D)
data_dependence2d$marker <- fix_marker_names(data_dependence2d$marker)
data_dependence2d$var_x <- fix_marker_names(data_dependence2d$var_x)
data_dependence2d$var_y <- fix_marker_names(data_dependence2d$var_y)
data_dependence2d$model <- gsub(" ", "_", data_dependence2d$model)

pdp2dmap <- list(
  "ALL" = list(
    cases = c(
      "Connectivity_Matrix,_MODL_256_tan--power_diag",
      "Cortical_Thickness--Connectivity_Matrix,_MODL_256_tan",
      "Cortical_Thickness--power_diag",
      "Cortical_Thickness--Subcortical_Volumes",
      "Subcortical_Volumes--Connectivity_Matrix,_MODL_256_tan",
      "Subcortical_Volumes--power_diag"),
    labels = c(
      "fMRI-P[cat]",
      "CrtT-fMRI",
      "CrtT-P[cat]",
      "CrtT-SbcV",
      "SbcV-fMRI",
      "SbcV-P[cat]")
  )
)

print(nrow(data_dependence2d))
print(unique(data_dependence2d$model))
models <- c("ALL")
for (i_model in seq_along(models)){
  this_model <- models[[i_model]]
  print(this_model)
  this_data <- subset(data_dependence2d,
                      model == this_model & model_type == 'rf_msqrt')

  #  make nicer marker labels for titles.
  this_data$marker_label <- this_data$marker
  for (i_case in seq_along(pdp2dmap[[this_model]][['cases']])) {
    this_data$marker_label <- gsub(
      pdp2dmap[[this_model]][['cases']][i_case],
      pdp2dmap[[this_model]][['labels']][i_case],
      this_data$marker_label)
  }

  marker_split <- strsplit(this_data$marker_label, '-')
  this_data$marker_label_x <- sapply(marker_split, `[[`, 1)
  this_data$marker_label_y <- sapply(marker_split, `[[`, 2)

  if(this_model == "ALL_MRI"){
    this_breaks <- seq(30, 80, 4)
  }else{
    this_breaks <- seq(46, 62, 2)
  }

  fig2e <- ggplot(data = this_data,
                  mapping = aes(x = x, y = y, z = pred)) +
      geom_raster(aes(fill = pred), show.legend = T) +
      stat_contour(breaks = this_breaks,
                  color = "white", bins = 7, show.legend = F) +
      scale_fill_viridis_c(breaks = this_breaks,
                          name = expression(hat(y)),
                          guide = guide_colorbar(barheight = 10)) +
      theme(panel.grid.major = element_blank(),
            panel.grid.minor = element_blank()) +
      facet_wrap(marker_label_x ~ marker_label_y,
                scales = "free",
                labeller = function(x) label_parsed(x, multi_line = F)) +
      xlab("Input age 1") +
      ylab("Input age 2")
  
  fname <- paste0("./figures/elements_fig2e_meg_pdp_2d",
                  "_", this_model, ".")
  ggsave(paste0(fname, "pdf"), plot = fig2e,
        width = save_width, height = save_height, useDingbats = T)
  embedFonts(file = paste0(fname, "pdf"), outfile = paste0(fname, "pdf"))
  ggsave(paste0(fname, "png"), plot = fig2e,
          width = save_width, height = save_height,
        dpi = 300)
}

#' ## Session info

#+ session_info
print(sessionInfo())
