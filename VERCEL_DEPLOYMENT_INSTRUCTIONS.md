# INSTRUCCIONES DETALLADAS DE DEPLOYMENT EN VERCEL

**Documento ubicado en:** `/Tik-Tok-Recommendation-System/VERCEL_DEPLOYMENT_INSTRUCTIONS.md`

**Guía complementaria:** `/Tik-Tok-Recommendation-System/DEPLOYMENT.md` (leer después de completar este documento)

---

## ESTADO ACTUAL DEL PROYECTO

✅ **Repositorio:** https://github.com/PredictiveSocialMedia/Tik-Tok-Recommendation-System  
✅ **Rama principal:** `main` (propia del repositorio)  
✅ **GitHub Pages:** ELIMINADO (no usar)  
✅ **Configuración Vite:** Estándar (lista para Vercel)  
✅ **Build:** Funciona correctamente ✓  
✅ **Dependencias:** Todas instaladas  

---

## PRE-REQUISITOS (HACER ANTES DE COMENZAR)

### 1. Crear cuenta Vercel
- Ir a https://vercel.com/signup
- Registrarse con GitHub (recomendado) para conectar repositorio automáticamente
- **Se necesita acceso a:** GitHub organization "PredictiveSocialMedia" o permisos en el repositorio

### 2. Permisos en GitHub
- Tener acceso al repositorio: https://github.com/PredictiveSocialMedia/Tik-Tok-Recommendation-System
- Mínimo: permisos de lectura
- Ideal: permiso para crear webhooks (para auto-deploy en cada push)

---

## PROCESO DE DEPLOYMENT (7 PASOS)

### PASO 1: CONECTAR REPOSITORIO A VERCEL
**Tiempo estimado:** 2 minutos

**Instrucciones paso a paso:**

1. Abrir en navegador: https://vercel.com/new
2. Buscar repositorio: Escribir "Tik-Tok-Recommendation-System" en la barra de búsqueda
3. Hacer clic en "Import" (botón azul)
   - **Debe verse:** El repositorio lleno de archivos y carpetas cargadas
4. Esperar a que Vercel auto-detecte la configuración:
   - **Repo name:** `Tik-Tok-Recommendation-System`
   - **Owner:** La organización/cuenta seleccionada
   - **Branch to deploy from:** `main` (IMPORTANTE: no cambiar)

**¿Qué NO cambiar en esta pantalla?** Todo debe detectarse automáticamente. Solo hacer clic en "Import".

---

### PASO 2: REVISIÓN DE CONFIGURACIÓN DE BUILD
**Tiempo estimado:** 2 minutos

Después de importar, Vercel mostrará pantalla "Configure Project". Verificar:

**Framework Preset:**  
- ✅ Debe mostrar: "Vite" (auto-detectado)
- Si no: seleccionar manualmente "Vite"

**Root Directory:**  
- ✅ Debe estar vacío o mostrar "."
- Esta es la raíz del repositorio, no `mvp-mock-ui/` (Vercel maneja la detección interna)

**Build Command:**  
- ✅ Debe mostrar: `npm run build`
- ✅ Ubicación del comando en código: `/mvp-mock-ui/package.json` línea 8
- NO cambiar

**Output Directory:**  
- ✅ Debe mostrar: `mvp-mock-ui/dist`
- Esto se calcula automáticamente desde `vite.config.ts` (línea 5-6)
- NO cambiar

**Install Command:**  
- ✅ Debe mostrar: `npm install`
- NO cambiar

**Node.js Version:**  
- ✅ Usar la recomendada por Vercel (heredada de `.node-version` o defaults)
- Mínimo requerido: Node.js 16+

### VERIFICACIÓN VISUAL

```
Framework detected: Vite ✅
Root Directory: . ✅
Build Command: npm run build ✅
Output Directory: mvp-mock-ui/dist ✅
Install Command: npm install ✅
```

Si todo se ve así → **Continuar a PASO 3**

---

### PASO 3: CONFIGURAR VARIABLES DE ENTORNO (SI NECESARIO)
**Tiempo estimado:** 3 minutos

**Este paso es OPCIONAL.** Solo hacer si la aplicación necesita APIs externas.

**Variables a configurar:**

| Variable | Valor | Ubicación del valor original |
|----------|-------|------------------------------|
| `DEEPSEEK_API_KEY` | Tu clave API | `/mvp-mock-ui/.env.local` línea 1 |
| `DEEPSEEK_MODEL` | (Generalmente: "deepseek-chat") | `/mvp-mock-ui/.env.local` línea 2 |
| `DEEPSEEK_BASE_URL` | (Generalmente: "https://api.deepseek.com") | `/mvp-mock-ui/.env.local` línea 3 |

**Cómo agregar en Vercel:**

1. En la pantalla "Configure Project", desplazarse hasta "Environment Variables"
2. Hacer clic en "Add" (botón para agregar nueva variable)
3. Para cada variable:
   - **Name:** (copiar el nombre exactamente como aparece en la tabla arriba)
   - **Value:** (copiar el valor de `.env.local`)
   - Hacer clic en "Add"
4. Repetir para cada variable

**IMPORTANTE:** Las variables en Vercel reemplazan automáticamente los valores de `.env.local`. No cambiar el archivo local.

**Si NO tienes claves API:** Dejar esta sección vacía y continuar.

---

### PASO 4: SELECCIONAR PROYECTO Y NOMBRE
**Tiempo estimado:** 1 minuto

1. **Team:** Seleccionar el equipo/organización donde vivirá el proyecto
   - Recomendado: "PredictiveSocialMedia" (si existe)
   - Alternativa: Tu cuenta personal

2. **Project Name:** El nombre que verá en https://vercel.com/dashboard
   - Sugerencia: `tiktok-recommendation-system` (en minúsculas, con guiones)
   - **NO importa el nombre exacto** (se puede cambiar después)

3. **Hacer clic:** "Deploy" (botón azul grande)

---

### PASO 5: ESPERAR DEPLOYMENT (BUILD AUTOMÁTICO)
**Tiempo estimado:** 3-5 minutos

Vercel ejecutará automáticamente:

```
npm install         ← Instala dependencias
npm run build       ← Compila TypeScript + Vite
                    ← Copia output a ./mvp-mock-ui/dist/
                    ← Sube a CDN global de Vercel
```

**Pantalla esperada:**
- Barra de progreso mostrando: "Preparing build"
- Luego: "Building"
- Finalmente: "✓ Ready. Deployed"

**Si falla en "Preparing build":**
- Ir a PASO 7 (Troubleshooting)

**Si falla en "Building":**
```
Error típico: "npm run build" failed
Solución: Ver PASO 7
```

---

### PASO 6: VERIFICAR DEPLOYMENT EXITOSO
**Tiempo estimado:** 2 minutos

**Después de ver "✓ Ready. Deployed":**

1. Hacer clic en "Visit" (link azul) o el botón "Open"
2. Deberá abrirse una URL como: `https://<project-name>.vercel.app`
3. **Esperar 5-10 segundos** a que la app cargue
4. Verificar:
   - ✅ La página carga (no hay error 404 o 500)
   - ✅ Se ve la interfaz con los controles
   - ✅ No hay errores rojos en consola del navegador (F12)

**Pantalla esperada:**
- Interfaz de UI morada/oscura
- Panel izquierdo para upload de video
- Área central para procesar
- Generalmente toma 2-3 segundos en cargar por primera vez

**Si algo no funciona → Ir a PASO 7 (Troubleshooting)**

---

### PASO 7: USAR APLICACIÓN (VERIFICACIÓN MANUAL)
**Tiempo estimado:** 5 minutos

**Pruebas básicas para confirmar que todo funciona:**

1. **Verificar carga de página:**
   - Abrir DevTools (F12 o Ctrl+Shift+I)
   - Ir a pestaña "Console"
   - No debe haber errores rojos
   - Hacer refresh (Ctrl+R) - la página recarga limpiamente

2. **Verificar que se carga data de demostración:**
   - La aplicación debe mostrar videos/datos predeterminados
   - Los datos vienen de: `/mvp-mock-ui/src/data/demodata.jsonl`
   - Si no se ven datos → Hay problema en la búsqueda de archivos

3. **Prueba interactiva (si aplica):**
   - Intentar hacer clic en botones
   - Intentar hacer scroll
   - La interfaz debe responder

**Resultado esperado:**  
✅ Sitio funciona completamente → **DEPLOYMENT EXITOSO**

---

## PASO 7: TROUBLESHOOTING (EN CASO DE ERRORES)

### ESCENARIO A: "Build failed" o "npm run build failed"

**Síntomas:**
- En Vercel dashboard se ve un ❌ rojo
- Mensaje: "Failed to build project"

**Causa probable:** Dependencias faltantes o error de compilación

**Solución:**

1. Abrir terminal en tu máquina local:
   ```powershell
   cd c:\Users\Juan Sebastian Peña\Desktop\BCSAI 3 YEAR\ChatBots\Tik-Tok-Recommendation-System\mvp-mock-ui
   npm install
   npm run build
   ```

2. Si falla localmente, el error dirá exactamente qué está mal.
   - Ejemplo: "Cannot find module 'xyz'" → Instalar con `npm install xyz`
   - Ejemplo: "Type error on line 45" → Revisar código de TypeScript

3. **Si falla en TypeScript:**
   - Archivo problemático: `/mvp-mock-ui/src/` (algún .ts o .tsx)
   - Revisar errores en: `npx tsc --noEmit` (solo comprueba, no compila)

4. Una vez arreglado localmente:
   ```powershell
   git add .
   git commit -m "fix: build issues"
   git push origin main
   ```

5. Vercel re-intentará automáticamente (esperar 1-2 minutos)

---

### ESCENARIO B: Build exitoso pero página muestra error 404

**Síntomas:**
- Vercel muestra "✓ Ready. Deployed"
- Pero al abrir el link: error 404

**Causa probable:** Output directory mal configurado

**Solución:**

1. Ir a project settings en Vercel:
   - Vercel dashboard → Seleccionar proyecto → Settings → General
   
2. Verificar: **Build & Deployment**
   - "Output Directory": debe ser `mvp-mock-ui/dist`
   - Si es diferente, cambiar y hacer clic en "Save"

3. En dashboard, hacer clic en "Redeploy" (arriba a la derecha)
   - Esperar 3-5 minutos

---

### ESCENARIO C: Build exitoso pero página en blanco o sin estilos

**Síntomas:**
- Página carga pero está completamente blanca
- O sin colores/estilos CSS

**Causa probable:** Base path incorrecto o assets no encontrados

**Solución:**

1. Verificar en navegador (F12 → Network):
   - Ver si los archivos `.js` y `.css` cargan (status 200)
   - Si muestran error 404 → problemas de ruta

2. Revisar `/mvp-mock-ui/vite.config.ts`:
   - **Línea 5:** `export default defineConfig({`
   - **NO debe tener:** `base: "./"`
   - **Debe verse:** Sin `base:` (ya que Vercel maneja automáticamente)
   - Documento actual ubicación: `/mvp-mock-ui/vite.config.ts`

3. Si está incorrecto, corregir localmente:
   ```powershell
   # Editar el archivo manualmente o ejecutar:
   cd c:\Users\Juan Sebastian Peña\Desktop\BCSAI 3 YEAR\ChatBots\Tik-Tok-Recommendation-System\mvp-mock-ui
   npm run build
   git add .
   git commit -m "fix: vite config for vercel"
   git push origin main
   ```

4. Esperar re-deploy automático

---

### ESCENARIO D: Imágenes de thumbnails no cargan

**Síntomas:**
- La página funciona pero no ve miniaturas de videos
- Muestra placeholders grises

**Causa probable:** Endpoint `/thumbnail` no disponible en Vercel

**Este es un problema conocido y esperado.** Soluciones:

**Opción 1 (Rápida - Ahora):**
- Usar una API proxy externa como: `https://cors-anywhere.herokuapp.com`
- Este es un problema de backend, no de frontend
- Reporte: Crear issue en GitHub para después

**Opción 2 (Mediano - 1-2 horas):**
- Implementar serverless function en Vercel (`/api/thumbnail.ts`)
- Requiere código en TypeScript
- Documentación: https://vercel.com/docs/serverless-functions/introduction

**Opción 3 (Largo - 4-8 horas):**
- Cargar URLs de thumbnails directamente de TikTok
- Requiere cambios en API/frontend
- Considerar CORS y seguridad

**Recomendación:** Para MVP, no es crítico. Reportar como upgrade futuro.

---

## RESUMEN DE ARCHIVOS CLAVE

Ubicaciones exactas de archivos importantes:

```
/Tik-Tok-Recommendation-System/
├── VERCEL_DEPLOYMENT_INSTRUCTIONS.md     ← ESTE ARCHIVO
├── DEPLOYMENT.md                          ← Guía rápida (complementaria)
├── mvp-mock-ui/
│   ├── vite.config.ts                     ← Configuración Vite (NO cambiar)
│   ├── tsconfig.json                      ← Configuración TypeScript (NO cambiar)
│   ├── package.json                       ← Scripts y dependencias (NO cambiar)
│   ├── .env.local                         ← Variables de entorno (privado)
│   ├── src/
│   │   ├── main.tsx                       ← Punto de entrada React
│   │   ├── data/demodata.jsonl            ← Datos de demostración
│   │   └── components/                    ← Componentes React
│   └── dist/                              ← Output generado (ignorar, auto-generado)
└── .git/                                  ← Control de versión (no tocar)
```

---

## CHECKLIST DE VERIFICACIÓN FINAL

Antes de declarar que el deployment es exitoso:

- [ ] Vercel dashboard muestra "✓ Ready. Deployed" (no ❌)
- [ ] URL del proyecto es accesible (ejemplo: `https://my-project.vercel.app`)
- [ ] Página carga sin errores 404 o 500
- [ ] Interfaz de usuario es visible (no página en blanco)
- [ ] Estilos CSS se ven correctamente (colores, layout)
- [ ] Datos de demostración se cargan (si la app muestra datos)
- [ ] DevTools Console (F12) no muestra errores críticos
- [ ] Proyecto está en https://vercel.com/dashboard (visible en listado)
- [ ] Auto-deploy está habilitado (opcional pero recomendado)

---

## CONTACTO Y SOPORTE

**Si algo falla:**

1. Revisar error exacto en Vercel dashboard (hacer clic en el deployment fallido)
2. Leer PASO 7 (Troubleshooting) above
3. Revisar archivo `/DEPLOYMENT.md` for additional tips

**Comandos útiles para debugging local:**

```powershell
# Navegar a carpeta
cd "c:\Users\Juan Sebastian Peña\Desktop\BCSAI 3 YEAR\ChatBots\Tik-Tok-Recommendation-System\mvp-mock-ui"

# Instalar dependencias (si falla build)
npm install

# Probar build localmente
npm run build

# Ver si hay errores TypeScript
npx tsc --noEmit

# Ejecutar en desarrollo (para testing local)
npm run dev
```

---

## TIMELINE ESTIMADO

| Paso | Duración | Acción |
|------|----------|--------|
| 1 | 2 min | Conectar repo a Vercel |
| 2 | 2 min | Revisar configuración |
| 3 | 3 min | Agregar variables (si aplica) |
| 4 | 1 min | Nombre del proyecto |
| 5 | 5 min | Vercel compila automáticamente |
| 6 | 2 min | Verificar que funciona |
| 7 | 5 min | Testing básico |
| **TOTAL** | **≈ 20 minutos** | **Deployment completo** |

---

## PREGUNTAS FRECUENTES

**P: ¿Qué significa "Vercel auto-detectó Vite"?**  
R: Vercel leyó los archivos del proyecto (package.json, vite.config.ts) y supo automáticamente cómo hacer build.

**P: ¿Por qué la URL es `*.vercel.app` y no un dominio personalizado?**  
R: Es el dominio por defecto. Se puede usar un dominio personalizado después en Settings.

**P: ¿Qué pasa si empujo cambios a `main` después del primer deploy?**  
R: Vercel detecta el push automáticamente y re-deploya en 2-5 minutos.

**P: ¿Puedo deployar desde otra rama?**  
R: Sí, pero está configurado para `main`. Para cambiar: Vercel Dashboard → Settings → Git → Branches.

**P: ¿El servidor Express en `server/index.ts` se despliega?**  
R: No automáticamente. Solo se despliega el código React. Para el servidor, necesita ser una Serverless Function o Backend separado.

---

## SIGUIENTES PASOS DESPUÉS DEL DEPLOYMENT

Una vez que está en vivo en Vercel:

1. **Agregue dominio personalizado** (opcional):
   - Vercel Settings → Domains
   - Apuntar DNS del dominio

2. **Configure auto-deploy** (recomendado):
   - Ya debería estar habilitado por defecto
   - Cada push a `main` → auto-redeploy

3. **Monitoree performance**:
   - Vercel Analytics (Settings → Analytics)
   - Revisar tiempos de carga

4. **Para la función de thumbnails**:
   - Crear issue en GitHub para implementar serverless function
   - Asignar a quien va a trabajar en backend

---

**Documento íntegramente preparado para pasar a otra persona o AI**  
**Última actualización:** 22 Feb 2026  
**Estado:** Proyecto listo para production

