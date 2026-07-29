# AGENTS.md — Project Rules

## Branch and Production Workflow

Use this promotion path for every change:

```text
feature/* → development → validation → main → production rebuild
```

- Create feature branches from `development`; merge completed features back into `development`.
- Validate `development` with compilation, parser/integration checks, and relevant live-service verification before promotion.
- `main` must match the exact production version. Never merge unvalidated work directly into `main`.
- After promoting `development` to `main`, push both branches and rebuild/restart production from a clean `main` checkout.
- Preserve Fuseki and RabbitMQ Docker volumes during deployment.
- Record or verify the deployed Git SHA; do not claim production alignment without evidence.
- Begin new feature work only when `development` and `main` accurately represent their environments.

## Change Discipline

- Keep feature commits focused; do not mix provider refactors with ingestion or release changes.
- Do not modify human-authored tests to make a change pass.
- Do not commit generated outputs, caches, credentials, or runtime logs.
- Update `CHANGELOG.md` for completed user-visible or architectural work.
