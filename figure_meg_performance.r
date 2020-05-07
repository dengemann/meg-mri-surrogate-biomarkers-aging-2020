#'---
#'title: "1.1 MEG performance"
#'author: "Denis A. Engemann"
#'date: "8/3/2019"
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

#+
library(ggplot2)
library(ggbeeswarm)
library(ggrepel)
library(pmr)

# imports `color_cats` and `get_label_from_marker`
source('./utils.r')
source('./config.r')

#' We'll first set up the data and configure plotting.
#' Also, in this analysis we'll focus on the scores, not the predictions.

#+ setup
DUMMY <- './data/age_stacked_dummy.csv'
PREDICTIONS <- './data/age_stacked_predictions_megglobal.csv'
SCORES <- './data/age_stacked_scores_megglobal.csv'
IMPORTANCES <- './viz_intermediate_files/age_stacked_importance_5_global.csv'
IMPORTANCES2 <- './viz_intermediate_files/age_stacked_importance_8.csv'

data_pred_wide <- read.csv(PREDICTIONS)
data_scores_wide <- read.csv(SCORES)

data_dummy <- read.csv(DUMMY)
data_dummy$MAE <- -data_dummy$MAE
data_scores_wide$chance <- data_dummy$MAE

names(data_scores_wide) <- fix_column_names(names(data_scores_wide))
names(data_pred_wide) <- fix_column_names(names(data_pred_wide))

# Move to long format.
reshape_names <- names(data_scores_wide)[!names(data_scores_wide) %in%
                                         c('repeat_', 'repeat_idx')]
data_scores <- reshape(data = data_scores_wide,
                       direction = 'long',
                       varying = reshape_names,  # values
                       v.names = 'MAE',
                       times = factor(reshape_names), # keys
                       timevar = 'marker',
                       new.row.names = NULL)

data_scores['is_meg'] <- grepl('MEG', data_scores$marker)
#'Remap the names to be more compact for plotting.



# geom_beeswarm(alpha=0.4, show.legend = F, cex=0.7, size = 1.5,
#               groupOnX=FALSE) 
#+ stack_labels
stacked_keys <- c(
  "MEG_handcrafted",
  "MEG_powers",
  "MEG_powers_cross_powers",
  "MEG_powers_cross_powers_handrafted",
  "MEG_cat_powers_cross_powers_correlation",
  "MEG_cat_powers_cross_powers_correlation_handcrafted",
  "MEG_cross_powers_correlation",
  "MEG_powers_cross_powers_correlation",
  "MEG_all"
)

stacked_labels <- c(
    'O',  # other
    'P',  # Powers (cat powers + single band powers + single band envelopes) 
    'P~XP[f]',  # Powers + Cross Powers
    'P~XP[f]~O',  # . + Others
    'P[cat]~XP[f]~C[f]', # cat powers (power  + diag) + cross powers + Correlations
    'P[cat]~XP[f]~C[f]~O',  # . + O
    'XP[f]~C[f]', # Cross powers + correlation
    'P~XP[f]~C[f]', # Pwoers + Cross Powers + Corr
    'P~XP[f]~C[f]~O'  # . + O
)

#' Let's first generate some statistical summaries

data_all_scores <- data_scores_wide[,c(stacked_keys, "chance")]

data_scores_meg <- data_all_scores

#+ fig1b
color_breaks <- c('Full', 'Combined Source', 'Source Activity', 'Source Connectivity', 'Sensor Mixed', 'Chance')
color_values <-setNames(
  with(color_cats, c(black, orange, vermillon, blue, `sky blue`, gray)),
  color_breaks
)
color_labels <- tolower(c(color_breaks[1:4], "sensor", "chance"))

#+ subset_meg_stack

# XXX idea, rank statistics
sub_sel <- c(1, 2, 7, 8, 9)
sel_meg <- c(stacked_keys[sub_sel], "chance")
model_ranking <- do.call(
  rbind, 
  lapply(seq_along(data_all_scores[,1]),
         function(ii) {
          out <- data.frame(
            rank = rank(data_all_scores[ii,sel_meg]),
            model = names(data_all_scores[,sel_meg]),
            fold = ii)
          rownames(out) <- NULL
          return(out)
        }))

model_ranking$family <- 'Combined Source'
model_ranking$family[
  model_ranking$model == 'MEG_powers'] <- 'Source Activity'
model_ranking$family[
  model_ranking$model == 'MEG_cross_powers_correlation'] <- 'Source Connectivity'
model_ranking$family[
  model_ranking$model == 'MEG_handcrafted'] <- 'Sensor Mixed'
model_ranking$family[
  model_ranking$model == 'MEG_all'] <- 'Full'
model_ranking$family[
  model_ranking$model == 'chance'] <- 'Chance'

model_ranking$family <- factor(model_ranking$family)

model_ranking$model <- factor(
  model_ranking$model,
  c("chance", stacked_keys[sub_sel]),
  c("Chance", stacked_labels[sub_sel]))


fig_rank_box <- ggplot(data = model_ranking,
       mapping = aes(x = rank, y = reorder(family, rank, mean), color = family, fill = family)) +
geom_boxplot(show.legend = T, alpha = 0.3) +
geom_jitter(show.legend = F, size=1.5, width = 0.12, alpha=0.5) +
scale_x_continuous(breaks = 1:7) +
scale_color_manual(values = color_values,
                   labels = color_breaks,
                   breaks = color_breaks, name = NULL) +
guides(color = guide_legend(nrow = 2, title.position = 'top'),
       fill = guide_legend(nrow = 2)) +
scale_fill_manual(values = color_values,
                  labels = color_breaks,
                  breaks = color_breaks, name = NULL) +
labs(x = 'Ranking across CV testing-splits', y = 'Stacking Models') +
theme(
      axis.text.y = element_blank(),
      legend.position = 'top',
      legend.justification = 'left',
      legend.text.align = 0)
print(fig_rank_box)


model_ranking$family <- factor(
  model_ranking$family,
  levels = rev(c("Chance", "Sensor Mixed", "Source Connectivity", "Source Activity", "Combined Source", "Full")),
  labels = rev(
      c("Chance", "Sensor\nMixed", "Source\nConnectivity", "Source\nActivity", "Combined\nSource", "Full")))

fname <- "./figures/elements_fig4_stacking_meg_supp_rank_box."
ggsave(paste0(fname, "pdf"),
       width = save_width, height = save_height, useDingbats = F)
embedFonts(file = paste0(fname, "pdf"), outfile = paste0(fname, "pdf"))
ggsave(paste0(fname, "png"),
       width = save_width, height = save_height, dpi = 300)
knitr::include_graphics(paste0(fname, "png"), dpi = 200)

##

rankmat <- data.frame(
  do.call(
    cbind,
    with(model_ranking,
      by(model_ranking, family, function(x) x$rank))))
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
  colnames(mat) <- levels(model_ranking$family)
  rownames(mat) <- levels(model_ranking$family)

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


fname <- "./figures/elements_fig4_stacking_meg_supp_rank_mat_pair."
ggsave(paste0(fname, "pdf"),
       width = save_width, height = save_height, useDingbats = F)
embedFonts(file = paste0(fname, "pdf"), outfile = paste0(fname, "pdf"))
ggsave(paste0(fname, "png"),
       width = save_width, height = save_height, dpi = 300)
knitr::include_graphics(paste0(fname, "png"), dpi = 200)


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

fmt1 <- "$M=%0.2f,SD=%0.2f,P_{2.5,97.5}=[%0.2f,%0.2f]$"
formats1 <- vector(mode="character", length=nrow(data_all_summary))
for(ii in seq_along(formats1)){
    this <- paste(
      data_all_summary$model[ii],
      do.call(sprintf,
            c(list(fmt1),
              with(data_all_summary[ii,], 
                   c(M, SD, X2.5., X97.5.)))),
      sep = '\t')
    formats1[[ii]] <- this
}
writeLines(
  formats1,
  './viz_intermediate_files/all_scores_meg_summary.txt'
)

data_dummy_diff <- data_all_scores - data_all_scores$chance

data_dummy_diff_summary <- summarize_table(data_dummy_diff)

data_dummy_diff_summary$Pr <- t(t(apply(data_dummy_diff, FUN = function(x) sum(x < 0), 2)))

knitr:::kable(data_dummy_diff_summary)


fmt2 <- "$Pr_{<Chance}=%0.2f%%,M=%0.2f,SD=%0.2f,P_{2.5,97.5}=[%0.2f,%0.2f]$"
formats2 <- vector(mode="character", length=nrow(data_dummy_diff_summary))
for(ii in seq_along(formats2)){
    this <- paste(
      data_dummy_diff_summary$model[ii],
      do.call(sprintf,
            c(list(fmt2),
              with(data_dummy_diff_summary[ii,],  c(Pr, M, SD, X2.5., X97.5.)))),
      sep = '\t')
    formats2[[ii]] <- this
}
writeLines(
  formats2,
  './viz_intermediate_files/all_scores_dummy_diff_meg_summary.txt'
)

#+ subset_meg
data_scores$prediction <- factor(ifelse(
  data_scores$marker %in% stacked_keys, 'stacked', 'linear'))

# For now ignore MRI
data_scores <- subset(data_scores, is_meg == T)
data_scores$marker <- factor(data_scores$marker)
data_scores$repeat_idx <- factor(data_scores$repeat_idx)

#'Now we compute best results, we need to aggregate by fold idx.

#+ agg_by_fold
data_scores_cv <- by(data_scores,
                     list(data_scores$repeat_idx,
                          data_scores$marker),
                     function(x){
                       data.frame(
                         marker = unique(x$marker),
                         repeat_idx = unique(x$repeat_idx),
                         cv_mean = mean(x$MAE),
                         cv_std = sd(x$MAE))
                     })
data_scores_cv <- do.call(rbind, data_scores_cv)

data_scores_cv$prediction <- factor(ifelse(
  data_scores_cv$marker %in% stacked_keys, 'stacked', 'linear'))

best_cv_mean_linear <- min(subset(data_scores_cv,
                                  prediction == 'linear')$cv_mean)
best_cv_mean_stacked <- min(subset(data_scores_cv,
                                   prediction == 'stacked')$cv_mean)

#'We are ready to plot the error distributions.

#+ fig1a
n_models <- by(data_scores,
               data_scores$prediction,
               function(x) data.frame(
                 marker = unique(x$prediction),
                 n_models =  length(unique(x$marker))))
n_models <- do.call(rbind, n_models)

labels_fig1 <- c(
  sprintf("%s (n=%d)", n_models[2,1], n_models[2,]$n_models),
  sprintf("%s (n=%d)", n_models[1, 1], n_models[1,]$n_models)
)

labels_fig1 <- setNames(labels_fig1, c('stacked', 'linear'))
colors_fig1 <- setNames(with(color_cats, c(black, `blueish green`)),
                        c('stacked', 'linear'))
data_scores$prediction <- relevel(data_scores$prediction, ref = "stacked")

fig1a <- ggplot(data = data_scores,
                mapping = aes(x = MAE,
                              color = prediction, fill = prediction)) +
  stat_density(trim = T, geom = 'area', size = 1, alpha = 0.3) +
  scale_x_log10(breaks = seq(0, 60, 5)) +
  scale_color_manual(
    values = colors_fig1,
    labels = labels_fig1) + 
  scale_fill_manual(
    values = colors_fig1,
    labels = labels_fig1) + 
  guides(
    color = guide_legend(nrow = 1, title.position = "left")) +
  theme(legend.position = c(0.4, 0.99)) +
  xlab("MAE (years)") +
  ylab("Density") +
  annotate(geom = "text", x = DUMMY_ERROR - 0.5, y = 5,
           label =  "predicting~bar(y)",
           parse = T,
           angle =90, size = annotate_text_size) +
  geom_vline(xintercept=DUMMY_ERROR, color = "black", linetype = 'dashed') +
  geom_rug(alpha = 0.5, mapping = aes(color = prediction, x = cv_mean),
           data = data_scores_cv)
print(fig1a)


fname <- "./figures/elements_fig1_error_distro."
ggsave(paste0(fname, "pdf"),
       width = save_width, height = save_height, useDingbats = F)
embedFonts(file = paste0(fname, "pdf"), outfile = paste0(fname, "pdf"))
ggsave(paste0(fname, "png"),
       width = save_width, height = save_height,
       dpi = 300)
knitr::include_graphics(paste0(fname, "png"), dpi = 200)

#'Remap the names to be more compact for plotting.

#'We can see that, across models, stacking stablizes and improves the results.
#'
#'Let's now look at the most interesting stacked models.
#'
#'However, we should first add some more description to the data.

#+ subset_meg_stack
data_meg_stacked <- subset(data_scores, prediction == 'stacked')

data_meg_stacked$family <- 'Combined Source'
data_meg_stacked$family[
  data_meg_stacked$marker == 'MEG_powers'] <- 'Source Activity'
data_meg_stacked$family[
  data_meg_stacked$marker == 'MEG_cross_powers_correlation'] <- 'Source Connectivity'
data_meg_stacked$family[
  data_meg_stacked$marker == 'MEG_handcrafted'] <- 'Sensor Mixed'
data_meg_stacked$family[
  data_meg_stacked$marker == 'MEG_all'] <- 'Full'

data_meg_stacked$family <- factor(data_meg_stacked$family)

data_meg_stacked$marker <- factor(
  data_meg_stacked$marker,
  stacked_keys,
  stacked_labels)

#+ fig1b

color_breaks <- c('Full', 'Combined Source',
                  'Source Activity', 'Source Connectivity',
                  'Sensor Mixed')
color_values <-setNames(
  with(color_cats, c(black, orange, vermillon, blue, `sky blue`)),
  color_breaks
)
color_labels <- color_breaks

sel_idx <- data_meg_stacked$marker %in% c(
  "O",
  "P",
  "XP[f]~C[f]",
  "P~XP[f]~C[f]",
  "P~XP[f]~C[f]~O"
)

if(FALSE){  # toggle to select everything
  sel_idx <- rep(T, nrow(data_meg_stacked))
}

data_meg_stacked_sel <- data_meg_stacked[sel_idx,]
data_meg_stacked_sel$marker <- factor(data_meg_stacked_sel$marker)
sort_idx <- order(by(data_meg_stacked_sel,
                     data_meg_stacked_sel$marker,
                     function(x) mean(x$MAE)))

fig1b <- ggplot(
  data = data_meg_stacked_sel,
  mapping = aes(y = MAE, x = reorder(marker, MAE, mean),
                color = family, fill = family)) +
  coord_flip() +
  geom_hline(yintercept=DUMMY_ERROR, color = "black", linetype = 'dashed') +
  geom_boxplot(show.legend = T, outlier.shape = NA, alpha = 0.5, size = 0.7) +
  stat_summary(geom = 'text',
               mapping = aes(label  = sprintf("%1.1f", ..y..)),
               fun.y= mean, size = 3.2, show.legend = FALSE,
               position = position_nudge(x=-0.49)) +
  geom_beeswarm(alpha=0.3, show.legend = F, size = 2.4) +
  guides(
    color = guide_legend(nrow = 3, title.position = "top")) +
  ylab("MAE (years)") +
  xlab("MEG stacking models") +
  annotate(geom = "text", y = DUMMY_ERROR - 0.3, x = 2.5,
           label =  "predicting~bar(y)",
           parse = T,
           angle = 90, size = annotate_text_size) +
  scale_color_manual(values = color_values,
                     labels = color_labels,
                     breaks = color_breaks, name = NULL) +
  scale_fill_manual(values = color_values,
                    labels = color_labels,
                    breaks = color_breaks, name = NULL) +
  guides(color = guide_legend(nrow = 2, title.position = 'top'),
         fill = guide_legend(nrow = 2)) +
  scale_y_continuous(breaks = seq(3, 20, 1)) +
  theme(
        axis.text.y = element_blank(),
        legend.position = 'top',
        legend.justification = 'left',
        legend.text.align = 0)
print(fig1b)

fname <- "./figures/elements_fig1_stacking_models."
ggsave(paste0(fname, "pdf"),
       width = save_width, height = save_height, useDingbats = F)
embedFonts(file = paste0(fname, "pdf"), outfile = paste0(fname, "pdf"))
ggsave(paste0(fname, "png"),
       width = save_width, height = save_height,
       dpi = 300)
knitr::include_graphics(paste0(fname, "png"), dpi = 200)

#'Now we investigate variable importance. We need to do some data massaging
#'in order to have everyhing in a neat structure.

#+ process_importance
stack_models <- c(
    'MEG-all-no-diag',
    'MEG all'
)

importance_methods <- c(
  'permutation',
  'mdi',
  'permutation_in'
)

# we'll expand the grid of options
importance_params <- expand.grid(stack_models, importance_methods)

data_importance <- read.csv(IMPORTANCES)
data_importance <- subset(data_importance, stack_model  == 'MEG all')

data_importance_perm <- subset(data_importance, imp_metric == 'permutation')
data_importance_mdi <- subset(data_importance, imp_metric == 'mdi')

data_importance_perm_in <- read.csv(IMPORTANCES2)
data_importance_perm_in <- subset(data_importance_perm_in, stack_model  == 'MEG all')



get_importance <- function(dat, data_scores = NULL) {

  # we have 1000 trees +the mean importance
  dat <- dat[, colSums(is.na(dat)) == 0]
  names(dat) <- fix_column_names(names(dat))
  # reshape
  stack_names <- Filter(function(x)!x %in% c(
                        "repeat_", "fold_idx",
                        "imp_metric", "mod_type", "index",
                        "stack_model", "X"), names(dat))
  dat <- reshape(
    data = dat, direction = "long", varying = stack_names,
    v.names = "importance", timevar = "marker", times = stack_names)
  dat$marker <- factor(dat$marker)
  dat$importance <- unlist(dat$importance)
  # prepare packaging output: 1) average importance, 2) linear prediction
  out <- data.frame(dat)
  out['stack_model'] <- unique(dat$stack_model)
  out['method'] <- unique(dat$mod_type)

  # make sure to only contrast with linear fits
  # and relevel factor after subsetting
  # XXX
  if(!is.null(data_scores)){
    print('Found scores. Attaching individual marker performance.\n')
    data_scores_lin <- subset(data_scores, prediction == 'linear')
    data_scores_lin$marker <- as.character(data_scores_lin$marker)
    # order by out marker
    out$marker  <- as.character(out$marker)
    # XXX by may not guarantee order ...
    unique_marker <- c()
    for (marker in out$marker){
      if(!marker %in% unique_marker){
        unique_marker <- c(unique_marker, marker)
      }
    }
    data_scores_lin <- do.call(rbind,
      lapply(unique_marker,
             function(x) {
               dd <- data_scores_lin[data_scores_lin$marker == x, c("MAE", "marker")]
               dd$marker <- x
               names(dd) <- c("cv_score", "marker")
               return(dd)
             })
    )
    stopifnot(out$marker == data_scores_lin$marker)
    out$cv_score <- data_scores_lin$cv_score
  }
  return(out)
}

importance_results_mdi <- get_importance(data_importance_mdi, data_scores)
importance_results_mdi$method <- 'mdi'
importance_results_perm <- get_importance(data_importance_perm, data_scores)
importance_results_perm$method <- 'permutation'

importance_results_perm2 <- get_importance(data_importance_perm_in)
importance_results_perm2$method <- 'permutation_in'

importance_results <- rbind(importance_results_mdi, importance_results_perm)

#+ enrich_importance_info

get_family_from_marker <- function(marker) {
  family <- rep('sensor mixed', length(marker))
  family[grepl('diag', marker)] <- 'source activity'
  family[grepl('corr', marker)] <- 'source connectivity'
  family[grepl('cross', marker)] <- 'source connectivity'
  return(factor(family))
}

get_variant_from_marker <- function(marker) {
  variant <- rep('base', length(marker))
  variant[grepl('power', marker)] <- 'signal'
  variant[grepl('envelope', marker)] <- 'envelope'
  variant[grepl('1_f', marker)] <- '1/f~slope'
  variant[grepl('MEG_aud', marker)] <- 'ERF~lat'
  variant[grepl('MEG_vis', marker)] <- 'ERF~lat'
  variant[grepl('MEG_alpha_peak', marker)] <- 'alpha~peak'

  return(factor(variant))
}

importance_results$family <- get_family_from_marker(importance_results$marker)
importance_results$variant <- get_variant_from_marker(importance_results$marker)

importance_results$marker_label <- get_label_from_marker(
  importance_results$marker)

importance_results$stack_model <- sub(
  "MEG all", "MEG_all", importance_results$stack_model)


importance_results_perm2$family <- get_family_from_marker(importance_results_perm2$marker)
importance_results_perm2$variant <- get_variant_from_marker(importance_results_perm2$marker)

importance_results_perm2$marker_label <- get_label_from_marker(
  importance_results_perm2$marker)

importance_results_perm2$stack_model <- sub(
  "MEG all", "MEG_all", importance_results_perm2$stack_model)

#' We are ready to plot our importance results.

#+ fig1c
color_labels <- c('source activity', 'source connectivity', 'sensor mixed')
color_breaks <- color_labels

color_values <-setNames(
  with(color_cats, c(vermillon, blue, `sky blue`)),
  color_breaks
)


importance_results$variant <- factor(importance_results$variant,
  levels = c("signal",  "envelope", "1/f~slope", "alpha~peak", "ERF~lat")
)

shape_breaks <- levels(importance_results$variant)
shape_values <- setNames(
  c(17, 25, 21, 22, 23),
  shape_breaks
)

method <- 'mdi'
stack_model <- 'MEG_all'

# lets dump the most important markers
percent_rank <- function(x) trunc(rank(x)) / length(x)
sub_data_agg <- aggregate(cbind(importance,cv_score) ~
                          marker + family + variant + marker_label * method,
                          importance_results, FUN = mean)

sub_data_agg2 <- aggregate(importance ~
                           marker + family + variant + marker_label * method,
                           importance_results_perm2, FUN = mean)

sub_data_agg2 <- merge(sub_data_agg2,
                       sub_data_agg[1:62,c('cv_score', 'marker')],
                       by = "marker", sort = F)

sub_data_agg <- rbind(sub_data_agg, sub_data_agg2)

# Write out in-sample permutation importance.

imp_table_sel <- c(2, 1, 3, 4, 6, 7)

write.csv(
  sub_data_agg2[order(percent_rank(sub_data_agg2$importance),
                      decreasing  = T),][1:10,imp_table_sel],
  file = paste0(paste('./viz_intermediate_files/importances', method, stack_model, sep = '_'),
                '.csv'))

writeLines(knitr::kable(
  sub_data_agg2[order(percent_rank(sub_data_agg2$importance),
                      decreasing  = T),][1:10,c(2, 1, 3, 4, 6, 7)],
  format = "latex"),
  "./viz_intermediate_files/importance_table.tex"
)

make_fig1c <- function(data){

  fig <- ggplot(
    data = data,
    mapping = aes(x = importance,
                  y = cv_score,
                  color = family,
                  fill = family,
                  shape = variant)) +

    geom_point(size = 3) +
    scale_shape_manual(
      values = shape_values,
      breaks = shape_breaks,
      labels = parse(text = shape_breaks),
      name = "input/feature") +
    scale_fill_manual(
      values = alpha(color_values, 0.2),
      labels = color_labels,
      breaks = color_breaks) +
    scale_color_manual(
      values = color_values,
      labels = color_labels,
      breaks = color_breaks) +
    guides(
      fill = F,
      color = guide_legend(
        nrow = 5, title.position = "top", order = 1),
      shape = guide_legend(
        nrow = 3, title.position = "top", order = 2)) +
    theme(
          legend.position = c(-0.1, 0.08),
          legend.justification = 'left',
          legend.box = "horizontal",
          legend.direction = 'horizontal') +
    scale_y_log10(breaks = seq(0, 50, 5)) +
    scale_x_log10(labels = function(x) format(x, scientific = FALSE)) +
    ylab("MAE (years)") +
    geom_label_repel(
      data = subset(data, percent_rank(data$importance) >= .9),
      aes(label = marker_label),
      parse = T,
      fill = 'white',
      force = 2,
      show.legend = F,
      segment.alpha = 0.4,
      box.padding = unit(0.35, "lines"),
      point.padding = unit(0.3, "lines")
    )
  return(fig)
}

for(method in importance_methods){
  fig <- make_fig1c(sub_data_agg[sub_data_agg$method == method,])
  
  if(method == "permutation"){
    fig <- fig + xlab("Variable importance (MAE)") +
      scale_x_log10(
        limits = c(0.0001, 1.1),
        breaks = c(0.0001, 0.001, 0.01, 0.1, 1),
        labels = function(x) format(x, scientific = FALSE))
  }else if(method == "permutation_in"){
    fig <- fig + xlab("Variable importance (MAE)") +
      scale_x_log10(
        labels = function(x) format(x, scientific = FALSE))
  }else if (method == 'mdi'){
    fig <- fig + xlab("Variable importance (%)") +
      scale_x_log10(labels = function(x) format(x * 1e2, scientific = FALSE))
  }

  print(fig)
  fname <- paste0("./figures/elements_fig1c_importance",
                  "_", method,
                  "_", stack_model, "_", method, ".")
  ggsave(paste0(fname, "pdf"), plot = fig,
         width = save_width, height = save_height, useDingbats = F)
  embedFonts(file = paste0(fname, "pdf"), outfile = paste0(fname, "pdf"))
  ggsave(paste0(fname, "png"), width = save_width, height = save_height,
         dpi = 300)
  knitr::include_graphics(paste0(fname, "png"), dpi = 200)
}

mdi <- subset(sub_data_agg, method == 'mdi')
perm <- subset(sub_data_agg, method == 'permutation')
perm2 <- subset(sub_data_agg, method == 'permutation_in')

cor1 <- cor.test(mdi[order(mdi$marker),]$importance,
                 perm[order(perm$marker),]$importance, method = 'spearman')
cor2 <- cor.test(mdi[order(mdi$marker),]$importance,
                 perm2[order(perm2$marker),]$importance, method = 'spearman')
cor3 <- cor.test(perm[order(perm$marker),]$importance,
                 perm2[order(perm2$marker),]$importance, method = 'spearman')

cor.details <- function(x){
  data.frame(
    p.value = ifelse(x$p.value == 0,
                     "2.2e-16",
                     sprintf("%e", x$p.value)),
    r2 = round(x$estimate ^ 2, 3),
    rho = round(x$estimate, 3),
    name = x$data.name
  )
}
cor.output <- do.call(rbind, lapply(list(cor1, cor2, cor3), cor.details))
write.csv(cor.output, "./viz_intermediate_files/corr_results_importance.csv")

#'Time to take a look at the partial dependencies.
#'We'll also enrich the data with descriptors for plotting, as above.

#+ read_dependence
OUT_DEPENDENCE_1D <- './data/age_stacked_dependence_model-full-1d.csv'
data_dependence_1d <- read.csv(OUT_DEPENDENCE_1D)
data_dependence_1d$marker <- factor(data_dependence_1d$marker)
data_dependence_1d$model_type <- factor(data_dependence_1d$model_type)
data_dependence_1d$model <- factor(data_dependence_1d$model)

#+ fig1d
data_dependence_1d$marker <- fix_marker_names(data_dependence_1d$marker)
data_dependence_1d$family <- get_family_from_marker(data_dependence_1d$marker)
data_dependence_1d$variant <- get_variant_from_marker(data_dependence_1d$marker)
data_dependence_1d$marker_label <- get_label_from_marker(
  data_dependence_1d$marker)

color_breaks_1d <- color_breaks[1:2]
color_labels_1d <- color_breaks[1:2]
color_values_1d <-setNames(
  with(color_cats, c(vermillon, blue)),
  color_breaks_1d
)

data_sel <- subset(data_dependence_1d,
  model_type == 'rf_msqrt' & model == 'MEG all')

data_sel_max <- aggregate(. ~ marker * family * variant * marker_label,
                          data = data_sel, FUN = max)

fig1d <- ggplot(data = data_sel,
       mapping = aes(x = value,  y = pred,
                     group = marker, color = family,
                     linetype = variant)) +
  geom_line(size = 1) + 
  xlab('Input age (years)') +
  ylab('Age prediciton (years)') +
  scale_color_manual(
    values = color_values_1d,
    labels = color_labels_1d,
    breaks = color_breaks_1d) +

  guides(
    color = guide_legend(
      nrow = 4, title.position = "top", order = 1),
    linetype = guide_legend(
      title = 'input', nrow = 4, title.position = "top", order = 2))+
    theme(
      legend.position = c(0.01, 0.65),
      legend.justification = 'left',
      legend.box = "vertical",
      ) +
  geom_label_repel(
    data = data_sel_max,
    aes(label = marker_label),
    parse = T,
    force = 2,
    show.legend = F,
    segment.alpha = 0.4,
    box.padding = unit(0.35, "lines"),
    point.padding = unit(0.3, "lines")
  )
print(fig1d)  

fname <- "./figures/elements_fig1d_dependence."
ggsave(paste0(fname, "pdf"), width = save_width,
       height = save_height, useDingbats = F)
embedFonts(file = paste0(fname, "pdf"), outfile = paste0(fname, "pdf"))
ggsave(paste0(fname, "png"), width = save_width,
       height = save_height, dpi = 300)
knitr::include_graphics(paste0(fname, "png"), dpi = 200)

#' ## Session info

#+ session_info
print(sessionInfo())
