# Faster renv startup for non-interactive R: the R language server times out
# after 3s. Side effect: non-interactive installs build from source.
if (!interactive()) {
    Sys.setenv(
        RENV_CONFIG_SYNCHRONIZED_CHECK = "FALSE",
        RENV_WATCHDOG_ENABLED = "FALSE",
        RENV_CONFIG_PPM_ENABLED = "FALSE"
    )
}
source("renv/activate.R")
