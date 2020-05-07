# Make figures for Engemann et al. MEG-fMRI Brian age.

all: fig1 fig2 fig3 fig4

fig1: figure_meg_performance.r
	r -e "rmarkdown::render('figure_meg_performance.r', output_format = 'all')"

fig2: figure_mri_fmri_meg.r
	r -e "rmarkdown::render('figure_mri_fmri_meg.r', output_format = 'all')"

fig3: figure_behavior.r
	r -e "rmarkdown::render('figure_behavior.r', output_format = 'all')"

fig4: figure_opp_learn.r
	r -e "rmarkdown::render('figure_opp_learn.r', output_format = 'all')"

clean:
	rm figure*.md