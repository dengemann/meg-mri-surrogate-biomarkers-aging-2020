library(ggplot2)

my_theme <- theme_minimal() + theme(
   text = element_text(family = 'Helvetica', size = 20),
   legend.text = element_text(size = 16),
   legend.title = element_text(size = 16),
   axis.text = element_text(size = 16)
)
theme_set(my_theme)
mini_theme <- function() {
  theme_minimal() + theme(
    text = element_text(family = 'Helvetica', size = 8),
    legend.text = element_text(size = 8),
    legend.title = element_text(size = 8),
    axis.text = element_text(size = 8))
}

annotate_text_size <- 6
save_width <- 7
save_height <- 5

color_cats <- list(
  "black" = "#242424",
  "orange" = "#EFA435",
  "sky blue" = "#3EB6E7",
  "blueish green" = "#009D79",
  "yellow" = "#F2E55C",
  "blue" = "#0076B2",
  "vermillon" = "#E36C2F",
  "violet" = "#D683AB",
  "gray" = "#808080"
)

LABELS_NPSYCH <- c(
  "ACER",
  "BentonFaces",
  "CardioMeasures[1]",
  "CardioMeasures[2]",
  "CardioMeasures[3]",
  "FluidIntelligence",
  "EkmanEmHex",
  "EmotionalMemory[1]",
  "EmotionalMemory[2]",
  "EmotionalMemory[3]",
  "EmotionRegulation[1]",
  "EmotionRegulation[2]",
  "EmotionRegulation[3]",
  "FamousFaces",
  "ForceMatching[1]",
  "ForceMatching[2]",
  "Anxiety",
  "Depression",
  "Hotel",
  "Hours",
  "MMSE",
  "MotorLearning[1]",
  "MotorLearning[2]",
  "PicturePriming[1]",
  "PicturePriming[2]",
  "PicturePriming[3]",
  "PicturePriming[4]",
  "Proverbs",
  "PSQI",
  "RT[1]",
  "RT[2]",
  "Synsem[1]",
  "Synsem[2]",
  "TOT",
  "VSTMcolour[1]",
  "VSTMcolour[2]",
  "VSTMcolour[3]",
  "VSTMcolour[4]"
)

# from script: age_prediction_dummy.py in camcan repo
DUMMY_ERROR = 15.469
