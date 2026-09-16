# Docker

## Run the dashboard

```sh
docker compose up --build
```

Open <http://localhost:8501>. The default dashboard uses the explicit synthetic
research data mode. Stop the service with `Ctrl+C`; use `docker compose down`
to remove the container.

The Compose configuration mounts `outputs/` so run summaries persist on the
host and mounts `data/raw/` read-only for locally managed source artifacts.
Raw or institutional data is intentionally excluded from the image build
context. Do not commit confidential extracts or credentials.

## Run a pipeline check

```sh
docker compose run --rm --entrypoint python dashboard run.py pipeline --n-accounts 100 --n-mc-sims 100
```

For an institutional portfolio, pass a path inside the container, for example
`--portfolio-path /app/data/raw/portfolio.csv`, and use
`--data-source institutional`.