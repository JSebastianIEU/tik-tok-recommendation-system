# Reporte Historico Formal del Repositorio

**Proyecto:** Tik-Tok-Recommendation-System  
**Fecha de corte:** 2026-02-23  
**Tipo de reporte:** Historico (sin proyeccion futura)

## 1) Alcance y metodologia

- Se uso historial Git completo disponible localmente con `git log --all`.
- Metricas por contributor calculadas con `git log --all --numstat` (adiciones/eliminaciones exactas por email de autor).
- Metricas de colaboracion calculadas por co-edicion historica de archivos.
- Estado tecnico validado con ejecucion real de:
  - `python -m pytest -q tests`
  - `npm run build` en `mvp-mock-ui`
- El analisis de ramas cubre las ramas presentes en este clone local.

## 2) Resumen ejecutivo historico

- **Commits totales:** 32
- **Rango temporal de commits:** 2026-01-26 -> 2026-02-22
- **Merge commits:** 5
- **Contributors unicos (por email):** 7
- **Archivos tocados historicamente (Git):** 112
- **Bus factor (50% de commits):** 2 personas

## 3) Distribucion temporal de actividad

| Mes | Commits |
|---|---:|
| 2026-01 | 2 |
| 2026-02 | 30 |

## 4) Actividad por rama

| Rama | Commits visibles |
|---|---:|
| main | 30 |
| origin | 30 |
| origin/main | 30 |
| origin/demodb | 27 |

**Divergencia `origin/main...origin/demodb`:**
- Exclusivos de `origin/main`: 5
- Exclusivos de `origin/demodb`: 2

## 5) Contribucion exacta por contributor (por email)

> Nota: esta seccion es exacta por identidad Git (email). Si una persona uso mas de un email, aparece en filas separadas.

| Contributor (email) | Nombre(s) observados | Commits | % Commits | Merges | Dias activos | +Lineas | -Lineas | Neto | % de +Lineas | Entradas `numstat` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| arielo_moreira@hotmail.com | Sambitz234 | 11 | 34.38% | 0 | 4 | 405 | 89 | 316 | 2.57% | 14 |
| jpenad.ieu2023@student.ie.edu | JSebastianIEU | 8 | 25.00% | 0 | 4 | 13517 | 2 | 13515 | 85.90% | 110 |
| 152277527+jsebastianieu@users.noreply.github.com | JSebastianIEU, Juan Sebastian Peña | 7 | 21.88% | 5 | 5 | 210 | 1 | 209 | 1.33% | 3 |
| farroseh2005@gmail.com | FaresQaddoumi | 3 | 9.38% | 0 | 2 | 50 | 17 | 33 | 0.32% | 5 |
| alparslan2.0@mac.home | Alp Arslan | 1 | 3.12% | 0 | 1 | 530 | 22 | 508 | 3.37% | 2 |
| 159762508+jadchebly@users.noreply.github.com | jadchebly | 1 | 3.12% | 0 | 1 | 484 | 46 | 438 | 3.08% | 5 |
| omekkawi.ieu2023@student.ie.edu | notabzdev | 1 | 3.12% | 0 | 1 | 539 | 14 | 525 | 3.43% | 4 |

**Totales de churn (`numstat`):**
- **+Lineas:** 15735
- **-Lineas:** 191
- **Neto:** 15544

## 6) Metricas de trabajo en equipo

- **Archivos de dueño unico:** 91 (81.25%)
- **Archivos multi-autor (2+):** 21 (18.75%)
- **Archivos con 3+ autores:** 1

### 6.1 Archivos mas colaborativos (por numero de autores)

| Archivo | #Autores historicos |
|---|---:|
| .github/workflows/ci.yml | 3 |
| tests/test_validation.py | 2 |
| tests/test_smoke.py | 2 |
| src/retrieval/search.py | 2 |
| src/retrieval/index.py | 2 |
| src/retrieval/README.md | 2 |
| src/research/sota_notes.md | 2 |
| src/research/experiments.md | 2 |
| src/research/README.md | 2 |
| src/data/README.md | 2 |
| src/common/validation.py | 2 |
| src/common/schemas.py | 2 |
| src/baseline/report.md | 2 |
| src/baseline/baseline_stats.py | 2 |
| scripts/validate_data.py | 2 |

## 7) Estado actual del proyecto al corte

### 7.1 Estado funcional verificado

- **Pipeline Python:** pruebas ejecutadas correctamente: `6 passed`.
- **Mock UI React/Vite:** build de produccion exitoso.
- **Servidor local mock (`mvp-mock-ui/server`):** endpoints de reporte/chat/thumbnail con fallback local cuando no hay provider.

### 7.2 Estado por componente

- **Mock UI (`mvp-mock-ui/`)**
  - Interfaz MVP operativa con flujo upload -> analisis mock -> reporte -> chat.
  - Modo estatico/mock-only disponible (fallback automatico).
- **Pipeline (`src/`, `scripts/`)**
  - Validacion de datos mock implementada.
  - Retrieval baseline TF-IDF implementado (`index.py`, `search.py`).
  - Baseline analytics implementado (`baseline_stats.py`) con reporte Markdown.
- **Webscraper**
  - **No implementado en este repo** al corte.
  - La documentacion del proyecto indica explicitamente que no incluye scraping en esta fase.

### 7.3 Hallazgos tecnicos historicos (observados)

- Mezcla de uso de APIs de Pydantic v1/v2 en codigo (se observaron warnings de deprecacion al correr tests).
- Desalineacion puntual entre `Makefile` y firma CLI en query script.
- Estado de working tree al corte: archivo `VERCEL_DEPLOYMENT_INSTRUCTIONS.md` aparece eliminado localmente sin commit.

## 8) Estructura del repositorio (snapshot)

### 8.1 Top-level

- `.github/`
- `data/`
- `docs/`
- `mvp-mock-ui/`
- `scripts/`
- `src/`
- `tests/`
- `.gitignore`
- `DEPLOYMENT.md`
- `Makefile`
- `README.md`
- `requirements.txt`

### 8.2 Distribucion de archivos propios (sin `.git` ni `node_modules`)

- **Total archivos propios detectados:** 112

| Extension | Cantidad |
|---|---:|
| .ts | 34 |
| .tsx | 30 |
| .py | 16 |
| .md | 11 |
| .svg | 4 |
| .json | 4 |
| .jsonl | 2 |
| .gitignore | 2 |
| .html | 2 |
| .css | 2 |
| .txt | 1 |
| [no_ext] | 1 |
| .js | 1 |
| .yml | 1 |
| .github-pages | 1 |

## 9) Documentacion Markdown revisada para este informe

- `README.md`
- `DEPLOYMENT.md`
- `docs/architecture/module_contracts.md`
- `mvp-mock-ui/README.md`
- `src/baseline/README.md`
- `src/baseline/report.md`
- `src/data/README.md`
- `src/retrieval/README.md`
- `src/research/README.md`
- `src/research/experiments.md`
- `src/research/sota_notes.md`

## 10) Cronologia historica de commits (completa al corte)

| Fecha | Hash | Autor | Email | Mensaje |
|---|---|---|---|---|
| 2026-02-22 | 3178272 | JSebastianIEU | jpenad.ieu2023@student.ie.edu | docs: add comprehensive Vercel deployment instructions |
| 2026-02-22 | d0da31c | JSebastianIEU | jpenad.ieu2023@student.ie.edu | docs: add Vercel deployment guide |
| 2026-02-22 | 00282f3 | JSebastianIEU | jpenad.ieu2023@student.ie.edu | chore: clean up GitHub Pages configuration, ready for Vercel deployment |
| 2026-02-21 | 3ac4718 | Juan Sebastian Peña | 152277527+JSebastianIEU@users.noreply.github.com | Merge pull request #7 from PredictiveSocialMedia/feat/github-pages-ready |
| 2026-02-21 | a2f4f3f | JSebastianIEU | jpenad.ieu2023@student.ie.edu | Prepare mvp-mock-ui for GitHub Pages static hosting |
| 2026-02-21 | 8e18acb | Sambitz234 | arielo_moreira@hotmail.com | more keyword and hastag expansions |
| 2026-02-21 | 556921c | Sambitz234 | arielo_moreira@hotmail.com | keyword  data |
| 2026-02-11 | 56a7fa5 | JSebastianIEU | jpenad.ieu2023@student.ie.edu | module contract |
| 2026-02-11 | 04c0eef | Juan Sebastian Peña | 152277527+JSebastianIEU@users.noreply.github.com | Merge pull request #4 from PredictiveSocialMedia/sprint1-research |
| 2026-02-11 | 5a12de2 | Juan Sebastian Peña | 152277527+JSebastianIEU@users.noreply.github.com | Merge pull request #6 from PredictiveSocialMedia/mlops/sprint1-ci |
| 2026-02-11 | 8c620ad | Juan Sebastian Peña | 152277527+JSebastianIEU@users.noreply.github.com | Merge pull request #5 from PredictiveSocialMedia/JadC-search_engine_baseline |
| 2026-02-11 | c11707f | FaresQaddoumi | farroseh2005@gmail.com | CI: lint tests only for sprint 1 |
| 2026-02-11 | 1375a6d | FaresQaddoumi | farroseh2005@gmail.com | Add smoke tests and verify mock data record count |
| 2026-02-10 | feecbd7 | Alp Arslan | Alparslan2.0@Mac.home | feat(sprint-1): add baseline analytics module |
| 2026-02-10 | 5dbf0da | jadchebly | 159762508+jadchebly@users.noreply.github.com | Implement baseline TF-IDF retrieval system |
| 2026-02-10 | bd2f3bf | FaresQaddoumi | farroseh2005@gmail.com | Add CI workflow, Makefile targets, and clean test lint |
| 2026-02-09 | 88055a4 | notabzdev | omekkawi.ieu2023@student.ie.edu | Add files via upload |
| 2026-02-07 | 6ec5966 | Juan Sebastian Peña | 152277527+JSebastianIEU@users.noreply.github.com | Merge pull request #3 from PredictiveSocialMedia/data_layer |
| 2026-02-07 | 5ae7044 | Sambitz234 | arielo_moreira@hotmail.com | testing ci |
| 2026-02-07 | 1ba34bd | Sambitz234 | arielo_moreira@hotmail.com | added pytest to requirements.txt |
| 2026-02-07 | af1959b | Sambitz234 | arielo_moreira@hotmail.com | added 25 more records to have 50 total mock records |
| 2026-02-06 | 62bbbab | Sambitz234 | arielo_moreira@hotmail.com | added basic validation unit tets for schema and validators |
| 2026-02-06 | cf545b8 | Sambitz234 | arielo_moreira@hotmail.com | edited readme describing mock post fields and constraints |
| 2026-02-05 | 36348f7 | Sambitz234 | arielo_moreira@hotmail.com | edited validate_data.py CLI to validate the JSON mock data |
| 2026-02-05 | e92f40b | Sambitz234 | arielo_moreira@hotmail.com | added additional validators |
| 2026-02-05 | a413b47 | Sambitz234 | arielo_moreira@hotmail.com | small changes in pydantic schemas for posts and submodels |
| 2026-02-05 | edff4e0 | Sambitz234 | arielo_moreira@hotmail.com | added initial mock data, 25 records |
| 2026-02-05 | c85aa22 | JSebastianIEU | jpenad.ieu2023@student.ie.edu | updated readme |
| 2026-02-05 | 8060bdb | JSebastianIEU | jpenad.ieu2023@student.ie.edu | ci.yml |
| 2026-02-05 | 5aca239 | JSebastianIEU | jpenad.ieu2023@student.ie.edu | initial instructions work division |
| 2026-01-28 | 6a691b2 | Juan Sebastian Peña | 152277527+JSebastianIEU@users.noreply.github.com | Fix formatting in README description |
| 2026-01-26 | e16efa5 | JSebastianIEU | 152277527+JSebastianIEU@users.noreply.github.com | Initial commit |

---

**Fin del reporte historico formal.**
