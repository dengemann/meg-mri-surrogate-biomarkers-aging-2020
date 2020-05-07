  #'---
#'title: "4.1 Opportunistic Learning Model"
#'author: "Denis A. Engemann"
#'date: "9/17/2019"
#'output:
#'    html_document:
#'        code_folding:
#'            hide
#'    md_document:
#'        variant:
#'            markdown_github
#'---

#+ config
library(ggplot2)
library(ggbeeswarm)
library(ggrepel)

# seed!
set.seed(42)

# imports `color_cats` and `get_label_from_marker`
source('./utils.r')
source('./config.r')

PREDICTIONS <- './data/age_stacked_predictions_megglobal.csv'
PREDICTIONS2 <- './data/age_stacked_predictions__na_coded.csv'
PREDICTIONS3 <- './data/age_stacked_predictions_meglocal.csv'

IMPORTANCES <- "./outputs/age_stacked_importance_8.csv"
data_pred_wide <- read.csv(PREDICTIONS)

#' Another important topic concersn opportunistic learning from the data
#' that is available despite missing data. The naive approach would be
#' to throw away any data point (subject) for which not all modalities
#' (MRI, fMRI, MEG) are present. Instead one can feature-code missingness and
#' let the random forest learn from missingness, if possible. Ensuing important
#' questions are:
#'
#' 1. Does an opportunistically trained model perform as well as a
#' conservatively one on the subsets of complete data?
#'
#' 2. Does an opportunistically trained model perform as well on the subsets
#' of partially complete data as a conservative model, trained only on that
#' data.
#'
#' The first one can be easily answered in the current stacking framework.
#' One simply has to remove the nans from the prediction and compare the
#' conservative model with the nan-coded model.
#'
#' The second question would require computing the local models that take all
#' data available for a given modality, retain the indices, and evaluate the
#' nan-coded model on the same indices. For fairness, one variant would require
#' to do unimodal stacking here.
#'
#' Let us investigate the first idea. If that one does not work we are in
#' trouble anyways.
#' Our first task is gonna be to make sure we understand in the NA-coded
#' dataset, which predictions were based on missing values.
#' It is important to understand that the computation was setup such that
#' in any case, the linear predictions contain nans and the union of all
#' subjects that had at least one modality or variable was retained.
#' What *is* different between the two datasets is the stacking columns,
#' which either contain nans or which do not contain nans. Let's make sure
#' all makes sense:

#+ load_nan_analysis

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
  "fMRI"
)

stacked_selection <- c(
  "ALL",
  "ALL_MRI",
  "ALL_no_fMRI",
  "MRI",
  "fMRI",
  "MEG_all"
)

data_pred_stacked_sel <- preprocess_prediction_data(
  df_wide = data_pred_wide, stack_sel = stacked_selection, drop_na = T)

data_pred_stacked_sel_nona <- preprocess_prediction_data(
  data_pred_wide, stacked_selection, drop_na = F)

data_pred_wide_na <- read.csv(PREDICTIONS2)
data_pred_stacked_sel_na <- preprocess_prediction_data(
  data_pred_wide_na, stacked_selection, drop_na = F)

#' let's now test that the numbers make sense.

#+ test_nan_index
n_total <- 674 # number of total camcan subjects 
n_global <- 536 # number of common subjects across modalities
n_missing_global <- n_total - n_global

n_repeats <- length(unique(data_pred_wide$repeat_idx))
# drop first idnex column temporarily
stopifnot(n_total - sum(is.na(rowSums(data_pred_wide[,-1]))) / n_repeats == n_global)
stopifnot(nrow(data_pred_stacked_sel_na) == nrow(data_pred_stacked_sel_nona))

# The na-coded case has predictions for all subjects
stack_cols <- grepl('stack', names(data_pred_wide))
stopifnot(
  sum(is.na(rowSums(subset(data_pred_wide_na,
                           repeat_idx == 0)[,stack_cols]))) == 0)
stopifnot(
  sum(is.na(rowSums(subset(
    data_pred_wide, repeat_idx == 0)[, stack_cols]))) == n_missing_global)

#' Now we need make sure that we really only have data for subjects with
#' at least one or two linear inputs.

#+ test_nan_index2
lin_col_idx <- xor(
  names(data_pred_wide_na) %in% c('age', 'repeat_idx', 'fold_idx', 'repeat.'),
  !stack_cols)

row_nans <- rowSums(is.na(subset(data_pred_wide_na)[, lin_col_idx]))
stopifnot(max(row_nans) < sum(lin_col_idx))
ggplot(data = data.frame(nans = row_nans[1:n_total]), mapping = aes(x = nans)) +
  geom_histogram() +
  scale_x_sqrt() +
  scale_y_sqrt(breaks = c(c(1, 10), seq(0, 100, 20), seq(100, 500, 50))) +
  labs(x = 'row-wise # NA', y = 'count [subjects]')
ggsave('./figures/elements_fig2_diagnostics_na_dist_rowise.png', dpi = 300)

#' Ok, we're all set. Interestingly there is a larger subgroup of subejcts
#' With more than 50 missing inputs. _We need to be careful here not to distort
#' the evaluation of performance with these subjects_.
#' Based on what we just saw, we can proceed as follows: 1) we get the global
#' missingness index from the stacked variables of the global dataset 2) we get
#' the modality-wise missingness index from the linear variables of any dataset.
#' With these, we can then subset the na-coded data and compare performance.

#+ na_codes_q1
good_mask_global <- !is.na(data_pred_stacked_sel_nona$pred)

data_opp_learn <- data_pred_stacked_sel_nona
data_opp_learn$MAE_na <- data_pred_stacked_sel_na$MAE
data_opp_learn$pred_na <- data_pred_stacked_sel_na$pred
data_opp_learn <-data_opp_learn[good_mask_global,]

# let's print a nice table. Which, with Knitr, becomse a kable ...

colors_opp_learn <- setNames(
  with(color_cats, c(black, orange, `blueish green`, blue, violet, vermillon)),
  c("MRI, fMRI, MEG", 'MRI, fMRI', 'MRI, MEG', 'MRI', 'fMRI', 'MEG'))

data_opp_learn$family <- factor(
  data_opp_learn$family,
  levels = c('Multimodal', 'MRI & fMRI', 'MRI & MEG', 'MRI', 'fMRI', 'MEG'),
  labels = c("MRI, fMRI, MEG", 'MRI, fMRI', 'MRI, MEG', 'MRI', 'fMRI', 'MEG'))

opp_learn_1_agg <- aggregate(
  cbind(MAE, MAE_na) ~ family, data = data_opp_learn, FUN = mean)
knitr::kable(opp_learn_1_agg)

fig2f <- ggplot(data = data_opp_learn,
                mapping = aes(x = pred, y = pred_na, color = family)) +
    geom_point(alpha = 0.3, show.legend = F) +
    facet_wrap(~family) +
    # reuse definitions from above.
    scale_color_manual(breaks = names(colors_opp_learn),
                       labels = names(colors_opp_learn),
                       values = colors_opp_learn) +
    scale_fill_manual(breaks = names(colors_opp_learn),
                      labels = names(colors_opp_learn),
                      values = colors_opp_learn) +
    scale_x_continuous(breaks = seq(20, 80, 10),
                       labels = seq(20, 80, 10)) +
    xlab("Age prediction (completed-modality cases)") +                  
    ylab("Age prediction (at least one modality)")
print(fig2f)

fname <- "./figures/elements_fig2f_supplement_na_coding"
ggsave(paste0(fname, ".pdf"), plot = fig2f,
        width = save_width, height = save_height)
ggsave(paste0(fname, ".png"), plot = fig2f,
          width = save_width, height = save_height,
        dpi = 300)
knitr::include_graphics(paste0(fname, ".png"), dpi = 200)

#' This looks like there is no evidence that the second opportunistic model
#' performed worse on the common subjects. Let us now approach the second
#' question: do modality-wise predictions suffer? For this we must consider
#' the outputs computed with then `local` option.
#' There won't be much to check this time, except dimensions.
#' The logic here boils down to, pushing over from each sub-model the missing
#' value indices to the nan-coded model.

#+ na_codes_q2
data_pred_wide_local <- read.csv(PREDICTIONS3)
data_pred_stacked_sel_local <- preprocess_prediction_data(
  data_pred_wide_local, stacked_selection, drop_na = F)

stopifnot(dim(data_pred_stacked_sel_local) == dim(data_pred_stacked_sel_na))
na_mask <- is.na(data_pred_stacked_sel_local$pred)

data_opp_learn2 <- data_pred_stacked_sel_na
names(data_opp_learn2)[names(data_opp_learn2) == "MAE"] <- 'MAE_na'
data_opp_learn2$pred[na_mask] <- NA
data_opp_learn2$MAE_local <- data_pred_stacked_sel_local$MAE
data_opp_learn2$pred_local <- data_pred_stacked_sel_local$pred


data_opp_learn2$family <- factor(
  data_opp_learn2$family,
  levels = c('Multimodal', 'MRI & fMRI', 'MRI & MEG', 'MRI', 'fMRI', 'MEG'),
  labels = c("MRI, fMRI, MEG", 'MRI, fMRI', 'MRI, MEG', 'MRI', 'fMRI', 'MEG'))

# let's print a nice table. Which, with Knitr, becomse a kable ...
opp_learn_2_agg <- aggregate(
  cbind(MAE_na, MAE_local) ~ family, data = data_opp_learn2, FUN = mean)
knitr::kable(opp_learn_2_agg)

fig2g <- ggplot(data = data_opp_learn2,
                mapping = aes(x = pred, y = pred_local, color = family)) +
    geom_point(alpha = 0.3, show.legend = F) +
    facet_wrap(~family) +
                # reuse definitions from above.
    scale_color_manual(breaks = names(colors_opp_learn),
                       labels = names(colors_opp_learn),
                       values = colors_opp_learn) +
    scale_fill_manual(breaks = names(colors_opp_learn),
                        labels = names(colors_opp_learn),
                        values = colors_opp_learn) +
    scale_x_continuous(breaks = seq(20, 80, 10),
                        labels = seq(20, 80, 10)) +
    xlab("Age prediction (locallly-completed cases)") +
    ylab("Age prediction (at least one modality)")
print(fig2g)

fname <- "./figures/elements_fig2g_supplement_na_coding."
ggsave(paste0(fname, "pdf"), plot = fig2g,
        width = save_width, height = save_height)
ggsave(paste0(fname, "png"), plot = fig2g,
          width = save_width, height = save_height,
        dpi = 300)
knitr::include_graphics(paste0(fname, "png"), dpi = 200)

data_opp_learn3 <- data_pred_stacked_sel_na
data_opp_learn3$MAE_full <- with(data_pred_stacked_sel_na,
  rep(MAE[marker == "ALL"], times = length(unique(marker))))
data_opp_learn3$pred_full <- with(data_pred_stacked_sel_na,
  rep(MAE[marker == "ALL"], times = length(unique(marker))))

data_opp_learn3$MAE_full[na_mask] <- NA
data_opp_learn3$pred_full[na_mask] <- NA
data_opp_learn3$MAE_local <- data_pred_stacked_sel_local$MAE
data_opp_learn3$pred_local <- data_pred_stacked_sel_local$pred

data_opp_learn3$family <- factor(
  data_opp_learn3$family,
  levels = c('Multimodal', 'MRI & fMRI', 'MRI & MEG', 'MRI', 'fMRI', 'MEG'),
  labels = c("MRI, fMRI, MEG", 'MRI, fMRI', 'MRI, MEG', 'MRI', 'fMRI', 'MEG'))

opp_learn_3_agg <- aggregate(
  cbind(MAE_full, MAE_local) ~ family, data = data_opp_learn3,
  FUN = mean)

knitr::kable(opp_learn_3_agg)

opp_learn_agg <- data.frame(
  family = factor(c(as.character(opp_learn_1_agg$family),
                    as.character(opp_learn_2_agg$family),
                    as.character(opp_learn_3_agg$family))),
  MAE_na = c(opp_learn_1_agg$MAE_na, opp_learn_2_agg$MAE_na,
             opp_learn_3_agg$MAE_full),
  MAE = c(opp_learn_1_agg$MAE, opp_learn_2_agg$MAE_local,
          opp_learn_3_agg$MAE_local),
  comparison = factor(rep(c("common",
                            "common extra",
                            "full vs reduced"),
                      each = 6))
)

fig4a <- ggplot(data = opp_learn_agg,
       mapping = aes(x = MAE, y = MAE_na)) +
  geom_point(
    size = 5,
    alpha = 0.8,
    fill = "white",
    stroke = 1,
    mapping = aes(shape = comparison, color = family)) +
  geom_line(size=0.5, mapping = aes(linetype = comparison)) +
  ylab(expression(MAE[opportunistic])) +
  xlab(expression(MAE[available])) +
  coord_fixed(ylim = c(4.5, 7), xlim =  c(4.5, 7)) +
  scale_linetype_manual(
    breaks = c("common", "common extra", "full vs reduced"),
    labels = c("common", "common extra", "full vs reduced"),
    values = c("solid", "dotted", "dashed"),
    name = "comparison") +
  scale_shape_manual(
    breaks = c("common", "common extra", "full vs reduced"),
    labels = c("common", "common extra", "full vs reduced"),
    values = c(21, 22, 23),
    name = "comparison") +
  scale_color_manual(breaks = names(colors_opp_learn),
                     labels = names(colors_opp_learn),
                     values = colors_opp_learn,
                     name = "stacking model") +
  scale_fill_manual(breaks = names(colors_opp_learn),
                    labels = names(colors_opp_learn),
                    values = colors_opp_learn,
                    name = "stacking model")

print(fig4a)

fname <- "./figures/elements_fig4a."
ggsave(paste0(fname, "pdf"), plot = fig4a,
        width = save_width, height = save_height, useDingbats = F)
embedFonts(file = paste0(fname, "pdf"), outfile = paste0(fname, "pdf"))
ggsave(paste0(fname, "png"), plot = fig4a,
        width = save_width, height = save_height,
        dpi = 300)
knitr::include_graphics(paste0(fname, "png"), dpi = 200)


#' There seems to be a third case to look for.
#' Let us look at how the performance of the full model, trained
#' opportunistically, compares with sub-models on their corresponding subsets
#' of cases. This means we need to repeat the predictions of our full-model
#' for each submodel, and then carry over the submodels nans.
#' Remember that we have consumed the folds, there are 10 predictions one for 
#' each repetition.

# aggregate by subject,  but modify function to return nans.
# Nay, we just use te na.action argument.
data_pred_wide_na_summ <- aggregate(
  . ~ X, data = data_pred_wide_na, FUN = mean, na.action = NULL)
#' na_coded3

#' now we have a dataset in which we can split by nan-group
#' time to move to tidy format.

data_pred_wide_na_summ$nan_group <- as.factor(sapply(
  seq(nrow(data_pred_wide_na_summ)),
  function(ii,
           these_names = names(data_pred_wide_na_summ),
           na_data = is.na(data_pred_wide_na_summ)) {
  sum(which(na_data[ii,]))
  }))

data_pred_wide_na_summ$nan_group_cnt <- sapply(
  data_pred_wide_na_summ$nan_group,
  function(nan_group, data = data_pred_wide_na_summ) {
    nrow(data[data$nan_group == nan_group,])
  }
)

data_pred_wide_na_summ$nan_type <- as.factor(sapply(
  seq(nrow(data_pred_wide_na_summ)),
  function(ii,
           these_names = names(data_pred_wide_na_summ),
           na_data = is.na(data_pred_wide_na_summ)) {
    no_stack <- !grepl("stacked", these_names)
    out_names <- these_names[!na_data[ii, no_stack]]
    out <- c()
    if ("Cortical.Thickness" %in% out_names |
        "Cortical.Surface.Area" %in% out_names |
        "Subcortical.Volumes" %in% out_names) {
      out <- c(out, 'aMRI')
    }
    if ("Connectivity.Matrix..MODL.256.tan" %in% out_names){
      out <- c(out, 'fMRI')
    }
    if (any(grepl("mne", out_names))) {
      out <- c(out, expression(MEG[src]))
    }
    if ("MEG.1.f.gamma" %in% out_names |
        "MEG.alpha_peak" %in% out_names |
        "MEG.1.f.low" %in% out_names |
        "MEG.aud" %in% out_names |
        "MEG.vis" %in% out_names |
        "MEG.audvis" %in% out_names) {
      out <- c(out, expression(MEG[sens]))
    }
    return(paste(out, collapse = ','))
}))

data_pred_wide_na_summ_long <- preprocess_prediction_data(
  data_pred_wide_na_summ, stacked_keys)


nan_types <- as.character(unique(data_pred_wide_na_summ_long$nan_type))
legend_nan_type <- setNames(
  c(
    expression(MEG[sens]),
    expression(MEG),
    expression(MRI~fMRI),
    expression(MRI~fMRI~MEG[sens]),
    expression(MRI~fMRI~MEG),
    expression(fMRI~MEG[sens])
  ),
  c(
    "MEG[sens]",
    "MEG[src],MEG[sens]",
    "aMRI,fMRI",
    "aMRI,fMRI,MEG[sens]",
    "aMRI,fMRI,MEG[src],MEG[sens]",
    "fMRI,MEG[sens]"
)
)

mae_by_nan_type <- aggregate(
  MAE ~ nan_type,
  data = subset(data_pred_wide_na_summ_long,
                marker == "ALL"),
  FUN = mean)

legend_nan_type <- legend_nan_type[order(mae_by_nan_type$MAE)]

group_counts <- aggregate(
  nan_group_cnt ~ nan_group_cnt : nan_group,
  data = subset(data_pred_wide_na_summ_long,
                marker == "ALL"),
  FUN = function(x) {x[[1]]})$nan_group_cnt

color_values <- setNames(viridisLite::viridis(6),
                         names(legend_nan_type))

fig4b <- ggplot(data = subset(data_pred_wide_na_summ_long,
                     marker == "ALL"),
                mapping = aes(
                y = MAE,
                color = nan_type,
                fill = nan_type,
                x = reorder(nan_type, MAE, mean))) +
  geom_beeswarm(cex = 0.4, size = 0.5, alpha = 0.5) +
  stat_summary(geom = "boxplot", fun.data = my_quantiles,
               alpha = 0.5, size = 0.7, width = 0.8) +
  stat_summary(geom = "errorbar", fun.data = my_quantiles,
               alpha = 0.5, size = 0.5, width = 0.5) +
  stat_summary(geom = 'text',
               mapping = aes(label = sprintf("%1.1f", ..y..)),
               fun.y = mean, size = 2.3, show.legend = FALSE,
               position = position_nudge(x = -0.5)) +
  coord_flip() +
  guides(
    color = F,
    fill = F,
    shape = F) +
  scale_fill_manual(
    breaks = names(color_values),
    values = color_values) +
  scale_color_manual(
      breaks = names(legend_nan_type),
      values = color_values) +
  xlab("Available inputs") +
  scale_x_discrete(labels = legend_nan_type) +
  theme(
        legend.position = 'right',
        legend.justification = 'left',
        legend.text.align = 0)
print(fig4b)

fname <- "./figures/elements_fig4b."
ggsave(paste0(fname, "pdf"), plot = fig4b,
        width = save_width, height = save_height, useDingbats = F)
embedFonts(file = paste0(fname, "pdf"), outfile = paste0(fname, "pdf"))
ggsave(paste0(fname, "png"), plot = fig4b,
          width = save_width, height = save_height,
        dpi = 300)
knitr::include_graphics(paste0(fname, "png"), dpi = 200)

#+ session_info
print(sessionInfo())