# :books: sandbox

Studies coded with R, Python, and [Quarto](https://quarto.org/), developed in a well-controlled environment (Docker, `renv`, `uv`) for reproducibility purposes.

The studies are published as a website at <https://konilo.github.io/sandbox/>.


## Setup and reproduction of the results

- Install [Docker](https://www.docker.com/), [VS Code](https://code.visualstudio.com/), and the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).
- Clone the repo and open it in VS Code.
```bash
git clone git@github.com:Konilo/sandbox.git
cd sandbox
code .
```
- Run "Dev Containers: Reopen in Container" from the command palette (Ctrl/Cmd + Shift + P). VS Code builds the image and runs [`.devcontainer/postCreateCommand.sh`](.devcontainer/postCreateCommand.sh), which installs TinyTeX and the project's R + Python dependencies.
- Once the container is ready:
  - `make preview`: serve the site locally with live reload.
  - `make render`: build the site to `_site/`.
