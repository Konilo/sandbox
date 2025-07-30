# This file only provides help for renv
# It attaches packages that are needed but are not used in other files, so that
# renv detects them and installs them automatically.

# Useful renv commands
renv::status()
?renv::status() # recommendations on how to manage different out-of-sync scenarios
renv::dependencies()
?renv::dependencies() # cf. .renvignore
renv::repair()
renv::snapshot()
?renv::snapshot() # cf. type arg
renv::clean()

# Installing boilerplate packages
renv::install("languageserver", prompt = FALSE)
renv::install("rlang", prompt = FALSE)
renv::install("jsonlite", prompt = FALSE)
install.packages(
    "vscDebugger",
    repos = "https://manuelhentschel.r-universe.dev"
)
renv::install("lintr", prompt = FALSE)
renv::install("utf8", prompt = FALSE)
install.packages(
    "https://p3m.dev/cran/latest/src/contrib/Archive/pillar/pillar_1.10.2.tar.gz",
    repos = NULL,
    type = "source"
)
renv::install("remotes", prompt = FALSE)
remotes::install_github("nx10/httpgd")

# Showing renv that those packages are needed
library(jsonlite)
library(rlang)
library(languageserver)
library(vscDebugger)
library(lintr)
