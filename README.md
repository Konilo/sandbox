# :books: sandbox

Studies coded with R, Python, and [Quarto](https://quarto.org/), developed in a well-controlled environment (Docker, `renv`, `uv`) for reproducibility purposes.


## Studies

- [Lifecycle Investing](https://github.com/Konilo/sandbox/tree/main/sandbox/lifecycle_investing/lifecycle_investing.pdf): an exploration of a few striking aspects of this approach in the context of retirement planning.
- [Artprice100](https://github.com/Konilo/sandbox/tree/main/sandbox/artprice100/artprice100.pdf): a basic financial analysis of the Artprice100 index.
- [Leveraged ETFs &ndash; The Case Of CL2](https://github.com/Konilo/sandbox/tree/main/sandbox/leveraged_etfs/leveraged_etfs.pdf): a study of daily-rebalanced leveraged ETFs and an outlook on CL2's relevance for long-term investors.
- [When to Migrate Away From an Expensive ETF to a Cheaper, Equivalent One?](https://github.com/Konilo/sandbox/tree/main/sandbox/etf_migration_breakeven/etf_migration_breakeven.pdf): determining when migrating from an ETF to another, equivalent ETF with a lower total expense ratio is financially advantageous.
- [Holding Cash or Investing in a Money Market Fund Between Trades?](https://github.com/Konilo/sandbox/tree/main/sandbox/cash_or_mmf_between_trades/cash_or_mmf_between_trades.pdf): determining when investing liquidities in a money market fund is more advantageous than holding them as cash during inter-trade periods.


## Setup and reproduction of the results

- Install [Docker](https://www.docker.com/), [VS Code](https://code.visualstudio.com/), and the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).
- Clone the repo and open it in VS Code.
```bash
git clone git@github.com:Konilo/sandbox.git
cd sandbox
code .
```
- Run "Dev Containers: Reopen in Container" from the command palette (Ctrl/Cmd + Shift + P). VS Code builds the image and runs [`.devcontainer/postCreateCommand.sh`](.devcontainer/postCreateCommand.sh), which installs TinyTeX and the project's R + Python dependencies.
- Once the container is ready, open a study and render ("preview") it (Ctrl/Cmd + Shift + K). The resulting document will be saved in the directory of the `qmd` file.
