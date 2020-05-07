#'function to nicely edit markers
get_label_from_marker <- function(marker, .) {
  g <- function(...) sub(..., fixed = T)
  marker_label <- g("MEG_", "", marker)
  marker_label <- g("mne_", "", marker_label)
  marker_label <- g("f_low", "f[low]", marker_label)
  marker_label <- g("ta_low", "ta[low]", marker_label)
  marker_label <- g("ma_lo", "ma[low]", marker_label)
  marker_label <- g("a_high", "a[hi]", marker_label)
  marker_label <- g("a_mid", "a[mid]", marker_label)
  marker_label <- g("envelope_corr_orth_", "orth~", marker_label)
  marker_label <- g("envelope_corr_", "", marker_label)
  marker_label <- g("envelope_diag_", "", marker_label)
  marker_label <- g("envelope_cross_", "", marker_label)
  marker_label <- g("envelope_cross", "", marker_label)
  marker_label <- g("power_diag_", "", marker_label)
  marker_label <- g("power_cross_", "", marker_label)
  marker_label <- g("power_cross", "", marker_label)
  marker_label <- g("envelope_diag", "E[cat]", marker_label)
  marker_label <- g("power_diag", "P[cat]", marker_label)
  marker_label <- g("1_f", "1/f", marker_label)
  return(factor(marker_label))
}

#'function to create family from marker.
get_family_from_marker <- function(marker) {
  family <- rep('other', length(marker))
  family[grepl('diag', marker)] <- 'power'
  family[grepl('corr', marker)] <- 'corr'
  family[grepl('cross', marker)] <- 'cross'
  return(factor(family))
}

#'function to create variant from marker.
get_variant_from_marker <- function(marker) {
  variant <- rep('base', length(marker))
  variant[grepl('power', marker)] <- 'power'
  variant[grepl('envelope', marker)] <- 'envelope'
  variant[grepl('1_f', marker)] <- '1/f'
  return(factor(variant))
}

#'function to fix column names from previous outputs.
fix_column_names <- function(col_names) {
  out_names <- gsub('...', '_', col_names, fixed = T)
  out_names <- gsub('..', '_', out_names, fixed = T)
  out_names <- gsub('.', '_', out_names, fixed = T)
  return(out_names)
}

#' function  to fix marker names, rowwise.
fix_marker_names <- function(marker) {
  out_marker <- gsub(' ', '_', marker, fixed = T)
  out_marker <- gsub('MEG_', '', out_marker, fixed = T)
  return(out_marker)
}

# function to make nicer axis labels from marker labels in pdp analysis
get_ax_name <- function(data, ax = 'x', abbr = F) {
  p <- paste0
  if (abbr == T) {
    abb <- abbreviate
  } else {
    abb <- function(x) x
  }
  name <- p(
    unique(data[[p(ax, "_label")]]),
    "(",
    abb(unique(data[[p(ax, "_variant")]])),
    "~",
    "','",
    "~",
    abb(unique(data[[p(ax, "_family")]])),
    ")"
  )
  return(parse(text = name))
}


#' Function to get prediction data in actionable form.
preprocess_prediction_data <- function(df_wide, stack_sel, drop_na = T,
                                       stacked_keys = NULL) {
  if(is.null(stacked_keys)){
    stacked_keys_ <- STACKED_KEYS
  }
  
  names(df_wide) <- fix_column_names(names(df_wide))
  non_data_cols <- c(
    "X", "repeat_idx", "age", "repeat", "fold_idx", "nan_group",
    "nan_group_cnt", "nan_type")
  reshape_sel <- Filter(function(x)!x %in% non_data_cols, names(df_wide))
  df_pred <- reshape(data = df_wide, direction = "long",
                     varying = reshape_sel, v.names = "pred",
                     times = reshape_sel, timevar = "marker")
  if (drop_na)
    df_pred <- na.omit(df_pred)

  df_pred['modality'] <- 'MRI'
  df_pred[grepl('MEG', df_pred$marker),]['modality'] <- 'MEG'
  df_pred[df_pred$marker == 'stacked_fMRI',]['modality'] <- 'fMRI'
  df_pred[grepl('Connectivity_Matrix',
                  df_pred$marker),]['modality'] <- 'fMRI'
  df_pred[grepl('ALL', df_pred$marker),]['modality'] <- 'Multimodal'

  df_pred$marker <- gsub("stacked_", "", df_pred$marker)

  df_pred$prediction <- factor(ifelse(
    df_pred$marker %in% stacked_keys_, 'stacked', 'linear'))

  df_pred$marker <- factor(df_pred$marker)
  df_pred['MAE'] <- abs(df_pred['age'] - df_pred['pred'])

  df_pred_stacked <- subset(df_pred, prediction == 'stacked')

  out <- df_pred_stacked[df_pred_stacked$marker %in% stack_sel,]
  out$family <- 'Multimodal'
  out$family[out$marker == 'ALL_MRI'] <- 'MRI & fMRI'
  out$family[out$marker == 'fMRI'] <- 'fMRI'
  out$family[out$marker == 'MEG_all'] <- 'MEG'
  out$family[out$marker == 'ALL_no_fMRI'] <- 'MRI & MEG'
  out$family[out$marker == 'MRI'] <- 'MRI'
  return(out)
}

STACKED_KEYS <- c(
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

#' function to process importance scores
get_importances <- function(data_importance, data_scores = NULL) {

  # we have 1000 trees +the mean importance
  data_importance <- data_importance[, colSums(is.na(data_importance)) == 0]
  names(data_importance) <- fix_column_names(names(data_importance))
  # reshape
  stack_names <- Filter(function(x)!x %in% c("is_mean", "mod_type", "index",
                        "stack_model", "X"), names(data_importance))
  data_importance <- reshape(
    data = data_importance, direction = "long", varying = stack_names,
    v.names = "importance", timevar = "marker", times = stack_names)
  data_importance$marker <- factor(data_importance$marker)
  data_importance$importance <- unlist(data_importance$importance)
  # prepare packaging output: 1) average importance, 2) linear prediction
  out <- subset(data_importance, is_mean == F)
  out <- do.call(rbind, by(out, out$marker, function(x) data.frame(
                           marker = unique(x$marker),
                           importance = mean(x$importance))))
  out['stack_model'] <- unique(data_importance$stack_model)
  out['method'] <- unique(data_importance$mod_type)

  # make sure to only contrast with linear fits
  # and relevel factor after subsetting
  if(!is.null(data_scores)){
    print('Found scores. Attaching individual marker performance.\n')
    data_scores_lin <- subset(data_scores, prediction == 'linear')
    data_scores_lin$marker <- factor(data_scores_lin$marker)
    data_scores_lin <- do.call(rbind,
      by(data_scores_lin, data_scores_lin$marker,
        function(x) data.frame(marker = unique(x$marker),
                               cv_score = mean(x$MAE))))
    mask <- data_scores_lin$marker %in% out$marker
    out$cv_score <- data_scores_lin[mask,]$cv_score
  }
  return(out)
}

my_quantiles <- function(x) {
  out <- setNames(quantile(x, probs = c(0.025, 0.25, 0.5, 0.75, 0.975)),
                  c("ymin", "lower", "middle", "upper", "ymax"))
  return(out)
}
