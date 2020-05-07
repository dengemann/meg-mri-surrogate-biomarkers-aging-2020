from surfer import Brain

print(__doc__)

brain = Brain("fsaverage", "both", "inflated", views="lateral",
              background="white")
"""
You can also use a custom colormap and tweak its range.
"""

brain.add_morphometry("thickness",
                      colormap="Blues", min=1, max=4,
                      colorbar=False)

brain.save_image('./figures/elements_fig_concept_anatomy.png')

