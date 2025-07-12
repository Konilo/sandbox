# This file only provides help for renv
# It attaches packages that are needed but are not used in other files, so that
# renv detects them and installs them automatically.

# Useful renv commands
renv::status()
?renv::status()
renv::repair()
renv::snapshot()
renv::dependencies()
renv::clean()
# renv::settings$snapshot.type("all")

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

# Showing renv that those packages are needed
library(jsonlite)
library(rlang)
library(languageserver)
library(vscDebugger)
library(lintr)
