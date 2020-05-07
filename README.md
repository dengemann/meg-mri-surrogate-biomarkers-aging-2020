# paper-brain-age-figures

Figure repo for Engemann et al 2020  brain age paper.


## Dependencies

Making the figures requires a:

1. the (non-committed) `./data` directory (obtained from the authors)
2. a recent Python install including (>=3.5), ideally Anaconda, with Pandas. For details, see file: 'python_requirements.txt'
3. ideally, a recent R Version (>= 3.0).

### R dependencies

I did my best to keep the R dependencies flat, *avoiding* the
[tidyverse](https://www.tidyverse.org) and other meta-packages. All logic and
control flow is written in conservative base R and avoids pipe operators and
other high-level syntax. The following visualization packages are needed to run
the code in a pure R console:

- [ggplot2](https://ggplot2.tidyverse.org)
- [ggbeeswarm](https://github.com/eclarke/ggbeeswarm)
- [ggrepel](https://github.com/slowkow/ggrepel)

To build the Markdown and HTML, you will also need:

- [rmarkdown](https://rmarkdown.rstudio.com)
- [knitr](https://cran.r-project.org/web/packages/knitr/index.html)

The dependencies themselves have rather flat dependencies. Ggplot,
may depend on a few elements of the tidyverse.
However, running the code here should be possible with a rather minimalistic
R setup.

## Running the code

The figure elements are created through R scripts, which at the same time
implement elements of [literate programming](https://en.wikipedia.org/wiki/Literate_programming)
through [RMarkdown](https://rmarkdown.rstudio.com) directives. 

The prinicipal R scripts begin follow the `figure_*.r` pattern and can be
run in the R console or can be compiled through Rmarkdown into HTML outputs.
In both cases, the figure elements are created and written to `./figures`.

To build the figures together with the HTML, please consider the Makefile.
You can build a single figure:

```bash
make fig2
```

Or all figures:

```bash
make all
```