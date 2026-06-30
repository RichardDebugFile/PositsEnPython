# Guía de contribución — Flujo de ramas (GitFlow)

Este proyecto usa **GitFlow** para mantener el orden de las ramas.

## Ramas permanentes

| Rama | Propósito | Recibe merges de |
|------|-----------|------------------|
| `main` | Código **estable / en producción**. Cada merge corresponde a una versión publicada y etiquetada. | `release/*`, `hotfix/*` |
| `develop` | Rama de **integración**. Contiene lo último ya terminado y probado. Base de las features. | `feature/*`, `release/*`, `hotfix/*` |

> No se hace commit directo a `main`. Tampoco a `develop` salvo merges.

## Ramas temporales

| Prefijo | Sale de | Vuelve a | Para qué |
|---------|---------|----------|----------|
| `feature/<descripcion>` | `develop` | `develop` | Nueva funcionalidad o mejora |
| `release/<x.y.z>` | `develop` | `main` **y** `develop` | Estabilizar y publicar una versión (tag en `main`) |
| `hotfix/<x.y.z>` | `main` | `main` **y** `develop` | Corrección urgente sobre producción |

Nombres en `kebab-case`. Ejemplos:
`feature/posits-redimensionables`, `release/2.6.0`, `hotfix/2.5.1`.

## Flujo de una feature

```bash
git checkout develop
git pull
git checkout -b feature/mi-mejora

# ...trabajo + commits...
pytest                      # los tests deben pasar antes de integrar

git checkout develop
git merge --no-ff feature/mi-mejora   # conserva el historial de la feature
git branch -d feature/mi-mejora
```

## Flujo de un release

```bash
git checkout -b release/2.6.0 develop
# ajustes finales, versión, changelog...
git checkout main && git merge --no-ff release/2.6.0
git tag -a v2.6.0 -m "Version 2.6.0"
git checkout develop && git merge --no-ff release/2.6.0
git branch -d release/2.6.0
```

## Flujo de un hotfix

```bash
git checkout -b hotfix/2.5.1 main
# arreglo + commit...
git checkout main && git merge --no-ff hotfix/2.5.1
git tag -a v2.5.1 -m "Hotfix 2.5.1"
git checkout develop && git merge --no-ff hotfix/2.5.1
git branch -d hotfix/2.5.1
```

## Convención de commits

- Mensaje en español, conciso, en imperativo (estilo del historial del repo).
- Asunto ≤ ~72 caracteres; cuerpo opcional para explicar el "por qué".

## Integración continua

El workflow de `.github/workflows/ci.yml` se ejecuta en cada `push` y `pull
request` hacia `main` y `develop`: `compileall` + `pylint` + `bandit` + `pytest`.
Una feature debería abrir PR hacia `develop` y pasar el CI antes de integrarse.
