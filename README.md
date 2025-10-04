# :books: sandbox

Studies coded with R, Python, and [Quarto](https://quarto.org/), developed in a well-controlled environment (Docker, `renv`, `uv`) for reproducibility purposes.


## Studies

- [Lifecycle Investing](https://github.com/Konilo/sandbox/tree/main/sandbox/lifecycle_investing/lifecycle_investing.pdf): an exploration of a few striking aspects of this approach in the context of retirement planning.
- [Artprice100](https://github.com/Konilo/sandbox/tree/main/sandbox/artprice100/artprice100.pdf): a basic financial analysis of the Artprice100 index.
- [Leveraged ETFs &ndash; The Case Of CL2](https://github.com/Konilo/sandbox/tree/main/sandbox/leveraged_etfs/leveraged_etfs.pdf): a study of daily-rebalanced leveraged ETFs and an outlook on CL2's relevance for long-term investors.
- [When to Migrate Away From an Expensive ETF to a Cheaper, Equivalent One?](https://github.com/Konilo/sandbox/tree/main/sandbox/etf_migration_breakeven/etf_migration_breakeven.pdf): determining when migrating from an ETF to another, equivalent ETF with a lower total expense ratio is financially advantageous.
- [Holding Cash or Investing in a Money Market Fund Between Trades?](https://github.com/Konilo/sandbox/tree/main/sandbox/etf_migration_breakeven/cash_or_mmf_between_trades.pdf): determining when investing liquidities in a money market fund is more advantageous than hold them as cash during inter-trade periods.


## Setup and reproduction of the results

- Install [Docker](https://www.docker.com/) and [VS Code](https://code.visualstudio.com/).
- Clone the repo.
```bash
git clone git@github.com:Konilo/sandbox.git
cd sandbox
```
- Open the cloned repo with VS Code.
```bash
code .
```
- Build the Docker image and start the container (use WSL if on Windows).
```bash
make start_container
```
- In the "Remote Explorer" section of VS Code's side bar, find the `sandbox:v1` container and attach to it.
- Now, inside the container, open the `/app/` dir where the repo lives.
- Install the [recommended extensions](https://github.com/Konilo/sandbox/blob/main/.vscode/extensions.json).
- Open a study and render ("preview") it (Ctrl/Cmd + Shift + K). The reulting document will be saved in the directory of the `qmd` file.
