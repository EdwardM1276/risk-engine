# Docker

## Run the dashboard

```sh
docker compose up --build
```

The build uses the latest tag contents available from the registry and applies
Debian security updates. Rebuild regularly rather than treating an old image
as patched:

```sh
docker compose build --pull --no-cache
docker compose up -d
```

Open <http://localhost:8501>. The default dashboard uses the explicit synthetic
research data mode. Stop the service with `Ctrl+C`; use `docker compose down`
to remove the container.

The Compose configuration mounts `outputs/` so run summaries persist on the
host and mounts `data/raw/` read-only for locally managed source artifacts.
Raw or institutional data is intentionally excluded from the image build
context. Generated CSV/JSON artifacts under `data/raw/` are ignored by Git;
the manifest may still be retained for provenance. Do not commit confidential
extracts or credentials.

## Run a pipeline check

```sh
docker compose run --rm --entrypoint python dashboard run.py pipeline --n-accounts 100 --n-mc-sims 100
```

For an institutional portfolio, pass a path inside the container, for example
`--portfolio-path /app/data/raw/portfolio.csv`, and use
`--data-source institutional`.

## Vulnerability checks

Build the image with a fresh base and scan the image itself (not just the
Dockerfile):

```sh
docker build --pull --no-cache -t risk-engine-dashboard:scan .
docker scout cves risk-engine-dashboard:scan
```

For CI or environments without Docker Scout, Trivy is an alternative:

```sh
trivy image --severity CRITICAL,HIGH --ignore-unfixed risk-engine-dashboard:scan
```

Dependency versions are bounded in `requirements.txt`; update them when an
advisory affects a transitive dependency, then rebuild with `--no-cache`.
Do not suppress a finding until the package, installed version, and fixed
version have been verified in the scanner output.