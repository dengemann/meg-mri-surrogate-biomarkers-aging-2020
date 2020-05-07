#'---
#'title: "3.1 Brain-age and behavior"
#'author: "Denis A. Engemann"
#'date: "9/1/2019"
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
library(boot)
library(memoise)

# imports `color_cats` and `get_label_from_marker`
source('./utils.r')
source('./config.r')

PREDICTIONS <- './data/age_stacked_predictions_megglobal.csv'
BEHAVIOR <- './behavioral_data/data_summary.csv'
PARTICIPANTS <- './data/participant_data.csv'
MOTION <-  './data/motion.csv'


#' We will first read in the behavioral data.

#+ get_behavioral_data
data_3000 <- read.csv('behavioral_data/homeint_3000.tsv', sep = '\t')
data_add <- read.csv('behavioral_data/additional_3000.tsv', sep = '\t')
data_self <- read.csv(
    'behavioral_data/self_completion_questionnaire_3000.tsv', sep = '\t')
data_npsych <- read.csv("data/neuropsych_scores.csv")
data_participants <- read.csv(PARTICIPANTS)
data_motion <- read.csv(MOTION)

names(data_npsych)[1] <- "Observations"

data_participants$hand[data_participants$hand == 0] <- NA
data_participants$hand_binary <- factor(
  ifelse(data_participants$hand > 0, "left", "right"))

#  aggregate
data_motion_agg <- do.call(rbind,
  with(data_motion,
    by(data_motion,
       subject,
       function (x, sel = c("x1","x2","x3","x4","x5","x6")) {
         data.frame(
          motion_norm = norm(as.matrix(x[,sel]), "F"),
          Observations = x$subject[1])
    }
)))

data_motion_agg$Observations <- factor(
  gsub("sub-", "", data_motion_agg$Observations))

data_behavior <- Reduce(
    function(x, y) merge(x, y, by = "Observations", all.x = T),
    list(data_3000, data_add, data_self, data_npsych, data_motion_agg,
         data_participants)
)
stopifnot(data_behavior$age.x == data_behavior$age.y)
data_behavior$age <- data_behavior$age.x
#' now the predictions

stacked_selection <- c(
  "ALL",
  "ALL_MRI",
  "ALL_no_fMRI",
  "MRI",
  "fMRI",
  "MEG_all"
)

data_pred_wide <- read.csv(PREDICTIONS)
data_pred_stacked_sel <- preprocess_prediction_data(
  df_wide = data_pred_wide, stack_sel = stacked_selection, drop_na = T)
names(data_pred_stacked_sel) <- sub(
    "X", "Observations", names(data_pred_stacked_sel))
names(data_pred_wide) <- sub(
    "X", "Observations", names(data_pred_wide))

data_behavior <- subset(data_behavior,
                        Observations %in% data_pred_wide$Observations)

age_data <- aggregate(age ~ Observations, data_pred_wide, unique)
data_behavior <- merge(data_behavior, age_data, by = "Observations")
stopifnot(sum(!data_behavior$age == age_data$age) == 0)


data_behavior <- within(data_behavior,
  {  # filte out implausible values.
    hours_in_bed[hours_in_bed > 20] <- NA
    hours_slept[hours_slept > 20] <- NA
    psqi[psqi > 20] <- NA
    HADS_anxiety[HADS_anxiety > 20] <- NA
    HADS_depression[HADS_depression > 20] <- NA 
    acer[acer > 500] <- NA
  }
)

# # this is what we may be interested in.
behavior_vars <- c(
 'psqi',  # not really continous (NRC), score
 'hours_slept', # count data
 'HADS_depression', # NRC, score
 'HADS_anxiety',  # NRC, score
 'acer', # NRC, score
 'mmse_i' # NRC, score
)

#' First thing to explore is a facet plot where each cell is one variable.
#' and scores are pitted against age predictions, grouped and coloured by
#' by brain age modality. The same thing is to be repeated after regressing out
#' age from any of these variables.
#'
#' In terms of preprocessing, we must compute the barin-age $\delta$
#' and also consider de-confounding the scores.
#'

data_pred_stacked_sel$delta <- with(data_pred_stacked_sel, pred - age)
data_pred_agg <- aggregate(delta ~ marker * Observations, data_pred_stacked_sel,
                           FUN = mean)

neuropsych_vars <- c(
  'BentonFaces',
  'CardioMeasures',
  'Cattell',
  'EkmanEmHex',
  'EmotionRegulation',
  'EmotionalMemory',
  'FamousFaces',
  'ForceMatching',
  'Hotel',
  'MotorLearning',
  'PicturePriming',
  'Proverbs',
  'RTchoice',
  'RTsimple',
  'Synsem',
  'TOT',
  'VSTMcolour')

# select sub-variables with same name root.
selection_ <- Reduce(
  c,
  lapply(
    neuropsych_vars,
    function(x) {
      mask <- grepl(x, names(data_behavior))
      return(names(data_behavior)[mask])
}))

selection <- c(
  "Observations",
  "age",
  "gender_text",
  "hand_binary",
  "motion_norm",
  selection_,
  behavior_vars
)

data_behavior_ <- data_behavior[,selection]
data_behavior_long <- reshape(
  data_behavior_,
  varying = c(selection_, behavior_vars),
  times = c(selection_, behavior_vars),
  direction = "long",
  timevar = "psych_marker",
  v.names = "value"
)

data_behavior_long$psych_marker <- factor(data_behavior_long$psych_marker)
data_behavior_long$psych_family <- factor(
  sapply(strsplit(as.character(data_behavior_long$psych_marker), split = "_"),
         "[[", 1))

rename_marker <- function(x){
  x <- sapply(strsplit(x, split = "_"),
            abbreviate)
  x <- sapply(x, function(x) paste(x, collapse = ' '))
  return(x)
}

new_names <- setNames(
    rename_marker(levels(data_behavior_long$psych_marker)),
    levels(data_behavior_long$psych_marker))

my_labeller <- labeller(psych_marker = new_names)

fig3a_supp1 <- ggplot(
  data = data_behavior_long,
  mapping = aes(x = age, y = value, color = psych_marker)) +
  geom_point(alpha = 0.1, size = 0.5, show.legend = F) +
  geom_smooth(show.legend = F) +
  facet_wrap(~psych_marker, scales = "free",
             labeller = my_labeller) +
  mini_theme()
print(fig3a_supp1)

fname <- "./figures/elements_fig3_supplement1_age_functions."
ggsave(paste0(fname, "pdf"), plot = fig3a_supp1,
       width = save_width * 1.3,
       height = save_height * 1.3, useDingbats = F)
embedFonts(file = paste0(fname, "pdf"), outfile = paste0(fname, "pdf"))
ggsave(paste0(fname, "png"),
       width = save_width * 1.3,
       height = save_height * 1.3,
       plot = fig3a_supp1, dpi = 300)
knitr::include_graphics(paste0(fname, "png"), dpi = 200)

data_behavior_long_dec <- do.call(rbind, by(
  data_behavior_long,
  data_behavior_long$psych_marker,
  function(df){
    out <- data.frame(
      value = rep(NA, nrow(df)),
      Observations = df$Observations,
      psych_family = df$psych_family,
      psych_marker = df$psych_marker,
      gender = df$gender_text,
      motion_norm = df$motion_norm,
      hand_binary  = df$hand_binary,
      age = df$age)
    good_mask <- !is.na(df$value)
    out$value[good_mask] <- resid(lm(value ~ poly(age, degree = 3),
                                  data = df[good_mask,]))
    return(out)
  }
))

fig3a_supp2 <- ggplot(
  data = data_behavior_long_dec,
  mapping = aes(x = age, y = value, color = psych_marker)) +
  geom_point(alpha = 0.1, size = 0.5, show.legend = F) +
  geom_smooth(show.legend = F) +
  facet_wrap(~psych_marker, scales = "free",
             labeller = my_labeller) +
  mini_theme()
print(fig3a_supp2)

fname <- "./figures/elements_fig3_supplement2_age_functions_deconfounded."
ggsave(paste0(fname, "pdf"), plot = fig3a_supp2,
       width = save_width * 1.3,
       height = save_height * 1.3, useDingbats = F)
embedFonts(file = paste0(fname, "pdf"), outfile = paste0(fname, "pdf"))
ggsave(paste0(fname, "png"),
       width = save_width * 1.3,
       height = save_height * 1.3,
       plot = fig3a_supp2, dpi = 300)
knitr::include_graphics(paste0(fname, "png"), dpi = 200)


data_full <- merge(data_pred_agg, data_behavior_long_dec, by = "Observations")
data_full2 <- merge(data_pred_agg, data_behavior_long, by = "Observations")
new_labels <- c('MRI, fMRI, MEG', 'MRI, fMRI', 'MRI, MEG','fMRI', 'MEG',  'MRI')

data_full$marker <- factor(data_full$marker)
levels(data_full$marker) <- new_labels
data_full2$marker <- factor(data_full2$marker)
levels(data_full2$marker) <- new_labels

colors_fig3a <- setNames(
  with(color_cats, c(black, orange, `blueish green`, blue, violet, vermillon)),
  c('MRI, fMRI, MEG', 'MRI, fMRI', 'MRI, MEG', 'MRI', 'fMRI', 'MEG'))


get_pval <- function(data, mod_fun){
  brain_age_pval <- do.call(rbind, by(
    data,
    list(data$psych_marker,
         data$marker),
    function(df) {
      df <- within(df, {
        value <- scale(value)
        delta <- scale(delta)
      })
      fit <- mod_fun(df)
      out <- data.frame(
        marker = factor(unique(df$marker)[1]),
        psych_marker = unique(df$psych_marker)[1],
        pval = summary(fit)$coefficients[2, 4],
        beta = coef(fit)[['delta']])
      return(out)
    }))

  new_names2 <- setNames(
      LABELS_NPSYCH,
      levels(data_behavior_long$psych_marker))
  levels(brain_age_pval$psych_marker) <- new_names2

  percent_rank <- function(x) trunc(rank(x)) / length(x)

  brain_age_pval <- within(brain_age_pval, {
    p_level <- rep('1', length(pval))
    p_level[pval <= 0.05] <- '2'
    p_level <- as.factor(p_level)
  })

  data_top_pval2 <- subset(
    brain_age_pval,
    p_level == 2
  )
  return(list(p = brain_age_pval, top = data_top_pval2))
}

brain_age_pval <- get_pval(data_full, function(df) lm(value ~ delta, df))

brain_age_pval2 <- get_pval(data_full2, function(df) lm(value ~ delta + poly(age, degree = 3), df))

brain_age_pval3 <- get_pval(data_full2, function(df) lm(
  value ~ delta + gender_text + hand_binary + log(motion_norm) + poly(age, degree = 3), df))


get_fig3b <- function(data, ymax = 6){
  pos <- position_jitterdodge(jitter.width = 1.7, seed = 42,
                              dodge.width = 0.2)

  fig3b <- ggplot(
    data = data$p,
    mapping = aes(y = -log10(pval),
                  x = reorder(marker, -log10(pval), FUN = max),
                  color = marker,
                  shape = p_level,
                  alpha = -log(pval),
                  size = -log10(pval))) +
    geom_point(position = pos, show.legend = F) +
    geom_text_repel(
      data = data$top,
      aes(label = psych_marker, size = -log10(pval)),
      parse = T,
      force = 2,
      position = pos,
      show.legend = F,
      segment.alpha = 0.4,
      size = 4
      ) +
    scale_alpha_continuous(range = c(0.05, 0.9), trans = 'sqrt', guide = F) +
    scale_shape_discrete(guide = F) +
    scale_color_manual(
      label = names(colors_fig3a),
      breaks = names(colors_fig3a),
      values = colors_fig3a) +
    scale_size_continuous(guide = FALSE) +
    guides(alpha = element_blank(), size = element_blank()) +
    geom_hline(yintercept = -log10(0.05)) +
    geom_hline(yintercept = -log10(0.01), linetype = 'dashed') +
    geom_hline(yintercept = -log10(0.001), linetype = 'dotted') +
    coord_cartesian(clip = 'off', ylim = c(-log10(1), ymax)) +
    xlab(element_blank()) +
    ylab(expression(-log[10](p))) +
    guides(color = guide_legend(
            title = element_blank(),
            position = "top", nrow = 2,
             title.position = "left")) +
    theme(
        axis.text = element_text(size = 14),
        legend.justification = 'left',
        legend.box = "horizontal")
  print(fig3b)
  return(fig3b)
}

models <- list(dec1 = brain_age_pval,
               dec2 = brain_age_pval2,
               dec3 = brain_age_pval3)

ymax <- list(dec1 = 5, dec2 = 6,dec3= 6)
for(method in names(models)){
  fig3b <- get_fig3b(data = models[[method]], ymax = ymax[[method]])

  fname <- paste0("./figures/elements_fig3b", "_", method, ".")
  ggsave(paste0(fname, "pdf"), plot = fig3b,
         width = save_width * 1.2,
         height = save_height, useDingbats = F)
  embedFonts(file = paste0(fname, "pdf"), outfile = paste0(fname, "pdf"))
  ggsave(paste0(fname, "png"),
         width = save_width * 1.2,
         height = save_height,
         plot = fig3b, dpi = 300)
  knitr::include_graphics(paste0(fname, "png"), dpi = 200) 
}


get_fig3c <- function(data){
  #' Idea: do horizontal coef plot this way. Use p-values for ggploting to
  #' indicate direction of effect. Change marker size /shape by sig level.
  pos <- position_jitterdodge(jitter.width = 1.7, 
                              dodge.width = 0.2, 
                              seed = 42)
  fig3c <- ggplot(
    data = data$p,
    mapping = aes(y = beta,
                  x = reorder(marker, - log10(pval), FUN = max),
                  color = marker,
                  group = marker,
                  alpha = -log(pval),
                  size = -log(pval),
                  shape = p_level)) +
    ylim(-.2, .2) +
    geom_point(position = pos, show.legend = F) +
    scale_color_manual(
      label = names(colors_fig3a),
      breaks = names(colors_fig3a),
      values = colors_fig3a) +
    ylab(expression(beta)) +
    xlab(element_blank()) +
    scale_size_continuous(guide = FALSE) +
    scale_alpha_continuous(range = c(0.05, 0.9), trans = 'sqrt', guide = F) +
    geom_text_repel(
      data = data$top,
      aes(label = psych_marker, size = -log10(pval)),
      parse = T,
      force = 2,
      position = pos,
      show.legend = F,
      segment.alpha = 0.4,
      size = 4
      ) +
    theme(
        axis.text = element_text(size = 14))
  print(fig3c)
}


for(method in names(models)){
  fig3c <- get_fig3c(data = models[[method]])

  fname <- paste0("./figures/elements_fig3c", "_", method, ".")
  ggsave(paste0(fname, "pdf"), plot = fig3c,
         width = save_width * 1.2,
         height = save_height, useDingbats = F)
  embedFonts(file = paste0(fname, "pdf"), outfile = paste0(fname, "pdf"))
  ggsave(paste0(fname, "png"),
         width = save_width * 1.2,
         height = save_height,
         plot = fig3c, dpi = 300)
  knitr::include_graphics(paste0(fname, "png"), dpi = 200) 
}

## Bootstrap time!

if(TRUE){
  mod_fun_dec <- function(data, idx) {
    coef(lm(value ~ delta, data[idx,]))[['delta']]
  }

  bootstrap_brain_age <- function(df, mod_fun) {

    boot_result <- boot(
        within(df, {
      value <- scale(value)
      delta <- scale(delta)
    }),
        mod_fun,
        R = 2000,
        parallel = "multicore",
        ncpus = 4)

    marker <- as.character(unique(df$marker)[1])
    psych_marker <- as.character(unique(df$psych_marker)[1])
    cat(marker, psych_marker, "\n")
    out <- data.frame(
          marker = marker,
          psych_marker = psych_marker,
          theta_boot = boot_result$t)
    out$theta <- boot_result$t0
    return(out)
  }

  memo_bootstrap_brain_age <- memoise(bootstrap_brain_age)

  brain_age_boot <- do.call(rbind, by(
    data_full,
    list(data_full$psych_marker,
         data_full$marker),
    function(x) memo_bootstrap_brain_age(x, mod_fun_dec)
  ))

  make_fig3a <- function(brain_boot){
    old_levels_ <- levels(brain_boot$marker)
    levels(brain_boot$marker) <- c(
      "MRI, fMRI, MEG", "MRI, fMRI", "MRI, MEG", "fMRI", "MEG", "MRI")

    levels(brain_boot$psych_marker) <- new_names

    fig3a <- ggplot(
      # data = subset(brain_boot, marker == "ALL"),
      data = brain_boot,
      mapping = aes(x = reorder(psych_marker, theta_boot, FUN = mean),
                    y = theta_boot,
                    fill = marker,
                    color = marker)) +
      coord_flip() +
      stat_summary(geom = "boxplot", fun.data = my_quantiles, show.legend = F,
                   alpha = 0.5, width = 0.8) +
      stat_summary(geom = "errorbar", fun.data = my_quantiles, show.legend = F) +
      geom_hline(yintercept = 0, color = 'black', linetype = 'dashed') +
      ylab(expression(beta[boot])) +
      xlab("Neuropsychological assessment") +
      theme(axis.text = element_text(size = 10)) +
      facet_wrap(~marker, nrow = 1) +
      scale_color_manual(values = colors_fig3a) +
      scale_fill_manual(values = colors_fig3a)
    print(fig3a)
    return(fig3a)
  }


  boot_models <- list(dec1 = brain_age_boot)

  if(FALSE){
    mod_fun_dec2 <- function(data, idx) {
      coef(lm(value ~ delta + poly(age, degree = 3), data[idx,]))[['delta']]
    }

    brain_age_boot2 <- do.call(rbind, by(
      data_full2,
      list(data_full$psych_marker,
           data_full$marker),
      function(x) memo_bootstrap_brain_age(x, mod_fun_dec2)
    ))
    boot_models$dec2 <- brain_age_boot2
  }

  for(model in names(boot_models)){
    fig3a <- make_fig3a(boot_models[[model]])
    fname <- paste0("./figures/elements_fig3a_", model ,"_.")
    ggsave(paste0(fname, "pdf"), plot = fig3a,
           width = save_width * 1.5,
           height = save_height * 1.5, useDingbats = F)
    embedFonts(file = paste0(fname, "pdf"), outfile = paste0(fname, "pdf"))
    ggsave(paste0(fname, "png"),
           width = save_width * 1.5,
           height = save_height * 1.5,
           plot = fig3a, dpi = 300)
    knitr::include_graphics(paste0(fname, "png"), dpi = 200)
  } 
}
#+ session_info
print(sessionInfo())