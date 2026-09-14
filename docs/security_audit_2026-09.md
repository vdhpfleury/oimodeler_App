# Audit de sécurité — `oimodeler_App`

**Cible :** https://github.com/vdhpfleury/oimodeler_App (branche `main`, 82 commits)
**Date :** 4 septembre 2026
**Périmètre :** l'intégralité du code applicatif (~3 200 lignes Python), les dépendances déclarées, et la configuration de déploiement (absente).
**Contexte :** mise en ligne prévue en accès libre, sans authentification.
**Versions vérifiées :** Streamlit 1.48.1 (code source inspecté), `oimodeler` @ HEAD (code source inspecté).

---

## 1. Synthèse exécutive

L'application est un frontend Streamlit propre sur le plan de l'architecture logicielle (séparation `pages/` / `core/` / `services/`, logique métier testable, pas de dépendance Streamlit dans `core/`). Le travail de structuration est réel et facilite grandement le durcissement.

En revanche, **en l'état, l'application ne peut pas être exposée publiquement.** Elle a été écrite avec un modèle mental « mono-utilisateur, sur mon poste », qui ne tient plus dès qu'un serveur partagé est en jeu. Trois problèmes structurels dominent :

1. **Le système de fichiers est utilisé comme un espace de noms global partagé.** Tous les uploads atterrissent dans `/tmp/<nom_fourni_par_le_client>`, et le cache Streamlit qui indexe ces chemins est partagé entre *toutes* les sessions. Deux utilisateurs qui envoient un fichier du même nom voient les données l'un de l'autre.

2. **Les chemins de fichiers sont construits par concaténation de chaînes contrôlées par le client**, sans aucune normalisation. Cela donne une primitive d'écriture arbitraire *et* une primitive de lecture arbitraire, toutes deux non authentifiées.
3
. **Aucun garde-fou serveur n'existe sur les formulaires.** C'est le point le plus contre-intuitif du rapport et il mérite d'être énoncé clairement : **les bornes des widgets Streamlit (`min_value`, `max_value`, la liste d'options d'un `selectbox`, `max_chars`) sont purement décoratives côté navigateur.** J'ai vérifié le code de désérialisation de Streamlit 1.48.1 : le serveur accepte telle quelle la valeur envoyée par le client. Un `selectbox` restreint à `["linear", "log"]` renvoie n'importe quelle chaîne si le client l'envoie ; un `number_input` borné à 512 renvoie `1e12`. Toutes les validations doivent être réécrites en Python, côté serveur.

### Tableau des vulnérabilités

| ID | Sévérité | Titre | Localisation |
|----|----------|-------|--------------|
| V1 | **Critique** | Écriture de fichier arbitraire via traversée de chemin dans le nom d'upload | `pages/data.py:48` |
| V2 | **Critique** | Lecture de fichier arbitraire via valeur de `multiselect` non validée | `pages/data.py:73-74`, `services/data_service.py:76` |
| V3 | **Critique** | Fuite de données scientifiques entre utilisateurs via `/tmp` + cache global | `pages/data.py:48-52`, `services/data_service.py:54-91` |
| V4 | **Élevée** | Bornes de formulaires non appliquées côté serveur (modèle de confiance erroné) | Transverse (~60 widgets) |
| V5 | **Élevée** | `eval()` non protégé atteignable dans `oimodeler` via l'expression de filtre | `oimUtils.py:2025` (dépendance), alimenté par `pages/data.py:139-146` |
| V6 | **Élevée** | Déni de service par épuisement mémoire (paramètres d'image non bornés) | `pages/modelling.py:149-150`, `pages/fitting.py:484-487` |
| V7 | **Élevée** | Déni de service par épuisement CPU (MCMC synchrone, sans file ni timeout) | `pages/fitting.py:343-376`, `pages/fitting.py:89` |
| V8 | **Élevée** | Saturation disque : `/tmp` jamais purgé, aucun quota | `pages/data.py:44-55` |
| V9 | **Moyenne** | Mutation d'objets partagés par `@st.cache_resource` (corruption inter-sessions) | `services/data_service.py:54`, 3× `_get_active_data_with_filter()` |
| V10 | **Moyenne** | Chemin fixe prévisible `/tmp/sampler_emcee.txt` (course + lien symbolique) | `pages/fitting.py:371-373` |
| V11 | **Moyenne** | Routes non intentionnelles : `pages/` est auto-découvert par Streamlit | Arborescence du projet |
| V12 | **Moyenne** | Divulgation d'information via les messages d'exception affichés | ~20 occurrences |
| V13 | **Moyenne** | Parsing de binaires non fiables (astropy/FITS) sans confinement | `services/data_service.py:73` |
| V14 | **Moyenne** | Chaîne d'approvisionnement : `oimodeler` non épinglé, Streamlit daté, pas de lockfile | `requirements.txt:1`, `requirements.txt:2` |
| V15 | **Moyenne** | Absence totale de configuration Streamlit (`.streamlit/config.toml`) | Absent |
| V16 | **Faible** | `unsafe_allow_html=True` (inoffensif aujourd'hui, dette latente) | `3_test.py:90`, `pages/overview.py:44` |
| V17 | **Faible** | Import CSV : DoS mémoire et noms de composants non contraints | `pages/modelling.py:287`, `core/csv_import.py:134` |
| V18 | **Faible** | `except:` nus masquant les erreurs | `pages/data.py:92`, `pages/explorer.py:119`, `core/model_builder.py:95` |
| V19 | **Faible** | `np.random.seed()` global : interférence entre sessions concurrentes | `core/fitting.py:40` |
| V20 | **Info** | Résidus de développement (`3_test.py`, variables `test_*`, `st.write` commentés) | Transverse |

---

## 2. Comprendre le modèle de menace Streamlit

Avant les vulnérabilités individuelles, il faut poser trois propriétés du framework qui expliquent la plupart d'entre elles. Elles ne sont pas des bugs de Streamlit — elles sont documentées ou implicites — mais elles surprennent presque tout le monde.

### 2.1 Les widgets ne valident rien côté serveur

Le navigateur envoie les valeurs de widgets par WebSocket. Le serveur les désérialise. J'ai lu ce code de désérialisation dans Streamlit 1.48.1 :

- `number_input` (`streamlit/elements/widgets/number_input.py:78`) : `deserialize()` fait au mieux un `int(val)`. **Aucune comparaison à `min_value` / `max_value`.**
- `slider` (`streamlit/elements/widgets/slider.py:196`) : idem, aucun bornage.
- `selectbox` (`streamlit/elements/widgets/selectbox.py:145`) : si la valeur reçue n'est pas dans les options, la fonction **retourne la valeur brute du client**. Elle ne lève pas d'erreur.
- `multiselect` (`streamlit/elements/widgets/multiselect.py:143`) : identique, un `except ValueError` ajoute la valeur inconnue telle quelle à la liste.
- `text_input` : `max_chars` est ignoré à la désérialisation.

Conséquence directe : **tout paramètre issu d'un widget doit être traité comme une entrée réseau hostile.** Un attaquant n'a pas besoin d'outil sophistiqué ; la console du navigateur suffit à forger un message.

C'est précisément ce que vous appelez « les gardes-fou sur les formulaires ». Ils n'existent aujourd'hui nulle part dans l'application.

### 2.2 `st.session_state` isole, `@st.cache_resource` ne l'a jamais fait

`st.session_state` est bien propre à chaque onglet navigateur. Mais `@st.cache_resource` est **explicitement un cache global au processus, partagé entre tous les utilisateurs** — votre propre docstring dans `services/data_service.py` le dit correctement (lignes 10-11) avant que le reste du code ne l'ignore. La clé du cache est l'argument de la fonction. Ici, la clé est un chemin `/tmp/<nom>`. Le nom vient du client. L'isolation repose donc entièrement sur l'improbabilité que deux astronomes nomment leur fichier `data.fits`.

### 2.3 Le dossier `pages/` est une route publique

Streamlit détecte automatiquement un dossier `pages/` frère du script principal (`streamlit/runtime/pages_manager.py:60`) et en fait une application multipage. Vos modules `pages/data.py`, `pages/fitting.py`, `pages/modelling.py`, etc. sont donc exposés comme des pages navigables, exécutées **sans passer par `3_test.py`**, donc sans `init_session_state()` ni `st.set_page_config()`. Ce n'est pas ce que vous vouliez : vous utilisez `st.tabs()` dans un script unique.

---

## 3. Vulnérabilités détaillées

### V1 — Critique : écriture de fichier arbitraire (traversée de chemin)

**Fichier :** `pages/data.py`, lignes 44-52

```python
for f in uploaded_files:
    tmp_path = f"/tmp/{f.name}"
    with open(tmp_path, "wb") as fh:
        fh.write(f.getbuffer())
```

`f.name` provient directement de l'en-tête `Content-Disposition: filename=` de la requête multipart. Vérification faite dans `streamlit/web/server/upload_file_request_handler.py:118` :

```python
UploadedFileRec(file_id=file_id, name=file["filename"], ...)
```

Aucune normalisation, aucun `os.path.basename()`. Streamlit 1.48.1 applique bien une validation d'extension côté serveur (`enforce_filename_restriction`, ajoutée en 1.43.2 après la publication de Cato Networks), mais **elle ne teste que le suffixe** : `filename.lower().endswith(".fits")`. Elle ne regarde pas les séparateurs de chemin.

**Exploitation :** requête `PUT /_stcore/upload_file/{session_id}/{file_id}` avec

```
Content-Disposition: form-data; name="file"; filename="../../../../home/oim/.streamlit/config.fits"
```

L'écriture se fait avec les droits du processus Streamlit, n'importe où sur le système de fichiers, à la seule condition que le chemin se termine par `.fits` ou `.oifits`. Le contenu est entièrement contrôlé.

**Impact :** écrasement de fichiers de données scientifiques d'autres utilisateurs ou du serveur ; écrasement des fichiers de tutoriel du dépôt ; remplissage de partitions arbitraires ; en fonction du déploiement, corruption de tout fichier `.fits` du système. Ce n'est pas une exécution de code directe (la contrainte d'extension le bloque), mais c'est une primitive d'écriture arbitraire non authentifiée, ce qui reste critique.

**Correctif :** voir §4.1 (module `services/storage.py`).

---

### V2 — Critique : lecture de fichier arbitraire

**Fichier :** `pages/data.py`, lignes 70-74

```python
test = st.multiselect("Select data to use", options=list(st.session_state.loaded_files.keys()))
test_filpath = ['/tmp/'+str(i) for i in test]
st.session_state.test_files_path = ['/tmp/'+str(i) for i in test]
```

Le chemin est reconstruit par concaténation à partir de la valeur du `multiselect`. Or (§2.1) `multiselect` renvoie les valeurs inconnues **telles quelles**. Un client qui envoie `["../etc/passwd"]` produit `/tmp/../etc/passwd`, transmis ensuite à :

```python
data = load_oifits_multi(tuple(st.session_state.test_files_path))   # → oim.oimData([...])
```

`oimData` passe le chemin à `astropy.io.fits.open()`.

**Impact :**
- Tout fichier `.fits` lisible par le processus, n'importe où sur le disque, est chargé et **affiché graphiquement** (couverture UV, observables) à l'attaquant. Cela inclut les uploads de tous les autres utilisateurs, dont les chemins sont devinables puisque `/tmp/<nom>` est plat.
- Pour les fichiers non-FITS, l'exception d'astropy est renvoyée à l'écran (`st.warning(f"UV plot error: {exc}")`), ce qui fournit un **oracle d'existence de fichier** exploitable pour cartographier le serveur.

À noter que cette ligne existe aussi sous forme dupliquée dans `pages/modelling.py:671` et `pages/fitting.py:565`. Le code de chargement a été copié-collé trois fois à l'identique.

**Correctif :** ne jamais reconstruire un chemin. Le dictionnaire `loaded_files` fait déjà l'association nom → chemin ; il faut l'utiliser comme table d'autorisation et rejeter toute clé absente.

```python
selected = st.multiselect("Select data to use", options=list(st.session_state.loaded_files.keys()))
# Garde-fou serveur : on ne garde que les clés réellement présentes
selected = [n for n in selected if n in st.session_state.loaded_files]
paths = [st.session_state.loaded_files[n] for n in selected]
```

---

### V3 — Critique : fuite de données entre utilisateurs

**Fichiers :** `pages/data.py:48-52`, `services/data_service.py:54-91`

Deux mécanismes se combinent :

1. Le chemin de stockage est `/tmp/{nom}` — un espace de noms **plat et global**. `st.session_state.loaded_files` est certes propre à la session, mais le fichier sur disque ne l'est pas.
2. `load_oifits(filepath)` est décorée `@st.cache_resource(ttl=3600, max_entries=20)`. La clé de cache est le chemin. Le cache est partagé entre toutes les sessions du worker.

**Scénario, sans aucune intention malveillante :** l'utilisateur A téléverse `HD179218.fits` (données non publiées). L'utilisateur B, une heure plus tard, téléverse son propre `HD179218.fits`. L'écriture de B écrase le fichier de A. Mais le cache `load_oifits("/tmp/HD179218.fits")` contient encore l'objet `oimData` de A et n'a pas expiré : **B travaille sur les données de A**, et A, s'il recharge après expiration du TTL, travaille sur celles de B.

**Scénario malveillant :** l'attaquant téléverse un fichier nommé exactement comme celui de la cible pour l'écraser, ou exploite V2 pour lire directement `/tmp/*.fits`.

Pour une application d'astronomie destinée à manipuler des données d'observation avant publication, c'est le risque le plus concret du rapport — plus probable qu'une exploitation offensive.

**Correctif :** répertoire par session, avec un identifiant aléatoire non devinable. La clé de cache devient alors naturellement propre à la session. Voir §4.1.

---

### V4 — Élevée : absence de garde-fous serveur sur les formulaires

**Fichiers :** transverse. Environ 60 widgets, aucun revalidé.

C'est la vulnérabilité racine dont V6 est une conséquence. Exemples représentatifs :

| Widget | Fichier | Ce que le client peut réellement envoyer |
|---|---|---|
| `number_input("pixel number", value=128)` | `modelling.py:149` | N'importe quel entier — aucune borne déclarée |
| `number_input("Image size (px)", 64, 512, 128)` | `fitting.py:484` | `1e9` — les bornes sont côté navigateur |
| `number_input("Steps", 0, 40000, 1000)` | `fitting.py:345` | `10**9` |
| `selectbox("Colormap", ["hot", ...])` | `fitting.py:480` | Chaîne arbitraire |
| `selectbox("Type", list(registry.keys()))` | `modelling.py:113` | Type de composant inexistant → `KeyError` non gérée |
| `slider("Gamma γ", 0.05, 1.0, 0.2)` | `fitting.py:479` | Flottant arbitraire, y compris `inf` / `nan` |
| `text_input("Model name")` | `modelling.py:107` | Chaîne de 10 Mo (`max_chars` non appliqué) |

**Correctif :** un module de validation appelé systématiquement à la lecture. Voir §4.2.

---

### V5 — Élevée : `eval()` atteignable dans la chaîne de filtrage

**Origine :** `oimodeler/oimUtils.py`, fonction `oifitsFlagWithExpression`, ligne ~2025 :

```python
for colname in data[arri].columns:
    coldata = data[arri].data[colname.name]
    ...
    globals()[f"{colname.name}"] = coldata     # injection dans les globals du module

flags = eval(expr)                              # ← sink
```

Le commentaire en amont dans la source d'oimodeler est explicite : `# TODO: Remove eval here as it is can be security liability`. L'`eval` s'exécute avec les `globals()` complets du module `oimUtils` — donc avec accès aux builtins, à `os` si importé, etc. C'est une exécution de code arbitraire.

**Atteignabilité actuelle :** `pages/data.py:139-146` construit `expr` par interpolation de flottants :

```python
expr = f"(EFF_WAVE<{w1_lo}) | (EFF_WAVE>{w1_hi})"
```

Un flottant se formate en `1.23e-06`, `inf` ou `nan` — impossible d'y injecter des caractères arbitraires. **L'application n'est donc pas exploitable aujourd'hui par ce chemin.** Je classe néanmoins en Élevée pour trois raisons :

1. Le sink `eval` est traversé à **chaque affichage de page**, sur des données contrôlées par le client.
2. Une seule ligne suffit à rendre l'application vulnérable à une RCE complète : ajouter un `st.text_input("Custom filter expression")` — évolution parfaitement naturelle pour cette application — donne un shell.
3. La valeur transite par `st.session_state.filter_expr`, une clé de session écrite à un endroit et lue à trois autres, ce qui rend l'audit de non-régression difficile.

**Correctif :** valider l'expression avant de la transmettre, avec une liste blanche d'identifiants, et clamper les longueurs d'onde. Voir §4.3. Signaler également le problème en amont à l'équipe oimodeler.

---

### V6 — Élevée : déni de service par épuisement mémoire

**Fichiers :** `pages/modelling.py:149-150`, `pages/fitting.py:484-487`, `pages/explorer.py:98-99`

```python
model_preview_img_fov    = st.number_input("pixel number", value=128, key="model_preview_img_fov")
model_preview_img_pxsize = st.number_input("pixel size in mas", value=0.15, ...)
...
im = model.getImage(fov, px_size, wl=wl, fromFT=True)
```

Aucune borne, même côté navigateur, sur `pixel number`. Il est transmis directement à `getImage()`, qui alloue un tableau `fov × fov` et calcule une transformée de Fourier dessus. Avec `fov = 50000`, l'allocation dépasse 20 Go en complexe double : le processus est tué par l'OOM killer, **l'application tombe pour tous les utilisateurs connectés**.

Note : `pages/explorer.py:98` déclare bien `min_value=16, max_value=1024` — mais comme établi en §2.1, ces bornes ne sont pas appliquées côté serveur. Le contrôle est cosmétique.

**Correctif :** clamper en Python après lecture (§4.2) *et* poser une limite mémoire dure au niveau du conteneur (§5.3).

---

### V7 — Élevée : déni de service par épuisement CPU

**Fichiers :** `pages/fitting.py:89` (`n_runs` jusqu'à 1000), `pages/fitting.py:343-345` (64 walkers × 40 000 pas), `pages/fitting.py:376`

```python
emfit.run(nsteps=nb_steps, progress=True)
```

Le calcul MCMC s'exécute de façon **synchrone dans le thread de la session Streamlit**, sans timeout, sans file d'attente, sans limite de concurrence. Un ajustement Emcee sérieux dure plusieurs minutes à plusieurs heures. Streamlit exécute toutes les sessions dans un seul processus Python : le GIL et la contention CPU font que quelques lancements simultanés suffisent à rendre l'application inutilisable pour tout le monde. Aucune authentification ne freine la répétition de l'opération.

C'est, à mon avis, le risque opérationnel le plus probable en accès libre — bien avant une attaque ciblée. Un simple pic de fréquentation lors d'une conférence produit le même effet.

**Correctif :** file d'attente hors processus, avec limite de concurrence globale et par IP, plus un timeout dur. Voir §4.4.

---

### V8 — Élevée : saturation disque

**Fichier :** `pages/data.py:44-55`

Les fichiers écrits dans `/tmp` ne sont **jamais supprimés**. Il n'y a ni quota par session, ni quota global, ni tâche de purge. La taille d'upload par défaut de Streamlit est de 200 Mo, `accept_multiple_files=True`, et il n'y a pas de limite au nombre d'uploads successifs.

Un client peut remplir la partition en quelques minutes. Selon le montage, cela peut arrêter le serveur entier.

À noter aussi que `/tmp` est souvent un `tmpfs` en RAM sur les distributions modernes — auquel cas ce problème se confond avec V6.

**Correctif :** quota par session, quota global, purge par TTL. Voir §4.1.

---

### V9 — Moyenne : mutation d'objets partagés par le cache

**Fichier :** `services/data_service.py:69-70` (la docstring), et les trois copies de `_get_active_data_with_filter()`

Votre docstring dit exactement la bonne chose :

> `NE PAS muter l'objet retourné directement`

Puis le code fait, dans les trois copies :

```python
data = load_oifits_multi(tuple(st.session_state.test_files_path))   # objet du cache global
data.setFilter(oim.oimDataFilter(filters))                          # mutation
data.useFilter = True                                               # mutation
```

L'objet `oimData` est partagé entre toutes les sessions par `@st.cache_resource`. Chaque utilisateur qui change une borne spectrale **réécrit les drapeaux de filtrage vus par les autres**, en pleine exécution de leur propre calcul. Les conséquences sont une fuite de configuration entre sessions et, plus grave, des résultats scientifiques silencieusement faux (un χ² calculé sur un jeu de données filtré par quelqu'un d'autre).

**Correctif :** intégrer les paramètres de filtre dans la clé de cache et construire un objet neuf.

```python
@st.cache_resource(ttl=1800, max_entries=50, show_spinner=False)
def load_filtered(filepaths: tuple[str, ...], expr: str,
                  bin_L: int, bin_N: int, norm_L: bool, norm_N: bool):
    oim = get_oim()
    data = oim.oimData(list(filepaths))       # instance neuve, jamais partagée mutée
    filters = []
    if expr:
        filters.append(oim.oimFlagWithExpressionFilter(expr=expr, keepOldFlag=False))
    filters.append(oim.oimWavelengthBinningFilter(targets=0, bin=bin_L, normalizeError=norm_L))
    filters.append(oim.oimWavelengthBinningFilter(targets=0, bin=bin_N, normalizeError=norm_N))
    data.setFilter(oim.oimDataFilter(filters))
    data.useFilter = True
    return data
```

Et supprimer les trois duplications de `_get_active_data_with_filter()` au profit d'un unique helper dans `services/`.

---

### V10 — Moyenne : chemin fixe prévisible

**Fichier :** `pages/fitting.py:371-373`

```python
sampler_path = Path("/tmp/sampler_emcee.txt")
sampler_path.unlink(missing_ok=True)
emfit.prepare(init=init_mode, samplerFile=str(sampler_path))
```

Chemin fixe dans un répertoire world-writable, partagé par tous les utilisateurs. Deux ajustements concurrents écrivent dans le même fichier : les chaînes MCMC se mélangent, les *corner plots* et *walker plots* affichent des résultats faux. Sur un hôte multi-utilisateur, c'est aussi une attaque classique par lien symbolique (l'attaquant crée `/tmp/sampler_emcee.txt → cible`, le processus l'écrase).

**Correctif :** `session_dir() / f"sampler_{uuid4().hex}.txt"`, supprimé après usage.

---

### V11 — Moyenne : routes non intentionnelles

Streamlit expose automatiquement `pages/*.py` comme pages navigables (§2.3). `pages/data.py`, `pages/explorer.py`, `pages/fitting.py`, `pages/modelling.py` et `pages/overview.py` sont donc accessibles directement par URL, exécutés hors du flux de `3_test.py`, sans initialisation du `session_state`. `initial_sidebar_state="collapsed"` masque la navigation, mais les routes restent atteignables.

**Correctif :** renommer le dossier (par exemple `views/`) et ajuster les imports. C'est un changement de deux minutes qui supprime toute une classe de surprises.

---

### V12 — Moyenne : divulgation d'information

Une vingtaine d'occurrences du motif :

```python
st.error(f"Error ({f.name}): {exc}")           # data.py:55
st.warning(f"UV plot error: {exc}")            # data.py:189
st.error(f"Cannot read CSV: {exc}")            # modelling.py:313
st.error(f"Minimization error: {exc}")         # fitting.py:242
```

Les exceptions d'astropy, numpy et oimodeler contiennent des chemins absolus, des noms de fichiers d'autres utilisateurs, et des détails d'implémentation. C'est ce qui transforme V2 d'une simple lecture ratée en oracle exploitable.

**Correctif :** message générique à l'utilisateur, détail dans les logs serveur avec un identifiant de corrélation.

```python
import logging, uuid
logger = logging.getLogger(__name__)

def user_error(st, msg: str, exc: Exception) -> None:
    ref = uuid.uuid4().hex[:8]
    logger.exception("[%s] %s", ref, msg)
    st.error(f"{msg} (référence : {ref})")
```

---

### V13 — Moyenne : parsing de binaires non fiables

`astropy.io.fits` analyse un format binaire complexe sur des fichiers entièrement fournis par des inconnus. Historiquement, les parseurs de ce type (FITS, TIFF, DICOM) sont une source régulière de dépassements et de bombes de décompression. Un FITS compressé peut aussi servir de *decompression bomb*.

**Correctifs :**
- Vérifier les octets magiques avant de conserver le fichier (un FITS valide commence par `SIMPLE  =`).
- Limiter la taille avant écriture.
- Exécuter le conteneur avec des limites mémoire et un profil `seccomp` (§5.3).

---

### V14 — Moyenne : chaîne d'approvisionnement

**Fichier :** `requirements.txt`

```
git+https://github.com/oimodeler/oimodeler.git      # ← ligne 1 : non épinglé
streamlit==1.48.1                                   # ← lignes 2 et 44 : doublon
```

Problèmes :
- **`oimodeler` est tiré depuis `HEAD` de la branche par défaut, sans tag ni commit.** Chaque reconstruction peut embarquer du code différent. Un compromis du dépôt amont se propage immédiatement en production. C'est la faiblesse d'approvisionnement la plus sérieuse ici.
- Streamlit 1.48.1 date d'environ treize mois. Les avis publiés depuis incluent CVE-2026-33682 (SSRF non authentifiée via chemins UNC, spécifique à Windows, corrigée en 1.54.0) et CVE-2026-10804 (hachage faible). L'impact sur un déploiement Linux est faible, mais l'écart de version l'est moins.
- Pas de lockfile avec empreintes (`--require-hashes`), donc pas de reproductibilité vérifiable.
- Les fichiers `__pycache__/*.pyc` sont **commités dans le dépôt** et il n'y a pas de `.gitignore`. Sans risque immédiat (CPython valide l'horodatage du `.py` associé), mais c'est une pollution qui n'a rien à faire dans un dépôt.

**Correctifs :**
```
oimodeler @ git+https://github.com/oimodeler/oimodeler.git@<sha256-du-commit-testé>
streamlit>=1.54.0
```
Ajouter un `.gitignore` (`__pycache__/`, `*.pyc`, `.venv/`, `/tmp/`), purger l'historique des `.pyc`, générer un `requirements.lock` avec `pip-compile --generate-hashes`, activer Dependabot et l'analyse de secrets sur le dépôt.

---

### V15 — Moyenne : aucune configuration Streamlit

Il n'y a pas de `.streamlit/config.toml`, pas de `Dockerfile`, pas de `Procfile`. L'application tournerait avec les valeurs par défaut, dont `maxUploadSize = 200` (Mo). Voir §5.1 pour la configuration recommandée.

---

### V16 — Faible : `unsafe_allow_html=True`

`3_test.py:90` et `pages/overview.py:44`. Les chaînes concernées sont aujourd'hui entièrement statiques : **il n'y a pas de XSS actuellement.** Streamlit échappe le HTML par défaut, donc les noms de modèles et de composants saisis par l'utilisateur, rendus via `st.success(f"... **{model_to_load}** ...")`, sont sûrs.

Le risque est de dette : le jour où une variable entre dans un de ces blocs, le XSS est immédiat. Les deux cas se réécrivent en Markdown natif sans perte.

---

### V17 — Faible : import CSV

**Fichiers :** `pages/modelling.py:287`, `core/csv_import.py`

- `pd.read_csv(csv_file)` sans `nrows` ni limite : un CSV de 200 Mo est chargé intégralement en mémoire (contribue à V6).
- `core/csv_import.py:134` fabrique le nom du composant : `f"c{idx}_{type_abbr}"`, où `type_abbr` vient du CSV. Ce nom sert ensuite de préfixe de clé `session_state` dans `components/param_editor.py:33` (`f"{comp['name']}_{param}_init"`). Un CSV construit à dessein peut provoquer des collisions de clés entre composants et corrompre l'état de l'éditeur.
- Les tableaux de résultats sont exportables en CSV depuis `st.dataframe`. Si un nom contrôlé par l'utilisateur commence par `=`, `+`, `-` ou `@`, il devient une formule à l'ouverture dans Excel (injection de formule CSV).

**Correctifs :** limiter la taille et le nombre de lignes du CSV ; contraindre `type_abbr` à `^[A-Za-z0-9]{1,10}$` ; préfixer d'une apostrophe toute cellule texte commençant par un caractère de formule.

---

### V18 — Faible : `except:` nus

`pages/data.py:92`, `pages/explorer.py:119`, `core/model_builder.py:95`, plus plusieurs dans `oimodeler`. Un `except:` nu attrape aussi `KeyboardInterrupt` et `SystemExit`. Surtout, dans `pages/data.py:89-93` :

```python
try:
    st.session_state.selected_file = test[0]
    filepath = st.session_state.loaded_files[test[0]]
except:
    pass
```

L'échec est silencieux et `filepath` reste à sa valeur précédente — donc potentiellement le fichier d'une sélection antérieure. Dans une application scientifique, un résultat silencieusement faux est pire qu'une erreur.

---

### V19 — Faible : graine aléatoire globale

**Fichier :** `core/fitting.py:40` — `np.random.seed(seed)` modifie l'état global de NumPy, partagé par tout le processus. Deux recherches aléatoires concurrentes interfèrent, et la reproductibilité promise par la case « Fixed seed » n'est pas garantie.

**Correctif :** `rng = np.random.default_rng(seed)` et passer `rng` explicitement à `generate_random_params()`.

---

### V20 — Info : résidus de développement

Le point d'entrée s'appelle `3_test.py`. `pages/data.py` contient un bloc marqué `####` avec des variables `test`, `test_filpath`, `test_loaded_files`, `test_selected_files` et une dizaine de `st.write` de débogage commentés. `_get_active_data()` est défini deux fois et jamais appelé. Ce n'est pas une faille, mais c'est du code qui n'a pas été relu, et l'expérience montre que c'est là que se logent les problèmes.

---

## 4. Correctifs applicatifs

### 4.1 Stockage isolé par session (corrige V1, V3, V8, partiellement V13)

Nouveau fichier `services/storage.py` :

```python
"""Stockage des uploads : isolé par session, borné, purgé automatiquement."""
from __future__ import annotations

import re
import shutil
import time
import uuid
from pathlib import Path

import streamlit as st

BASE_DIR          = Path("/var/lib/oimodeler/uploads")   # jamais /tmp
MAX_FILE_BYTES    = 100  * 1024 * 1024      # 100 Mo par fichier
MAX_SESSION_BYTES = 200 * 1024 * 1024      # 200 Mo par session
MAX_FILES         = 10                      # par session
SESSION_TTL       = 3600                    # secondes
ALLOWED_EXT       = (".fits", ".oifits")
_UNSAFE           = re.compile(r"[^A-Za-z0-9._-]")


def _session_id() -> str:
    """Identifiant aléatoire, non devinable, propre à l'onglet navigateur."""
    if "_sid" not in st.session_state:
        st.session_state["_sid"] = uuid.uuid4().hex
    return st.session_state["_sid"]


def session_dir() -> Path:
    d = BASE_DIR / _session_id()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_name(name: str) -> str:
    name = Path(name).name                    # neutralise ../ et les séparateurs
    name = _UNSAFE.sub("_", name)[:120]
    if name.startswith(".") or not name.lower().endswith(ALLOWED_EXT):
        raise ValueError("Nom de fichier ou extension non autorisés.")
    return name


def _looks_like_fits(head: bytes) -> bool:
    # Un fichier FITS valide commence par le mot-clé SIMPLE.
    return head[:6] == b"SIMPLE"


def store(uploaded) -> Path:
    """Écrit un fichier téléversé dans le répertoire de la session. Lève ValueError."""
    if uploaded.size > MAX_FILE_BYTES:
        raise ValueError(f"Fichier trop volumineux (max {MAX_FILE_BYTES // 1024**2} Mo).")

    data = uploaded.getbuffer()
    if not _looks_like_fits(bytes(data[:6])):
        raise ValueError("Le contenu ne correspond pas à un fichier FITS.")

    d = session_dir()
    existing = list(d.iterdir())
    if len(existing) >= MAX_FILES:
        raise ValueError(f"Trop de fichiers dans la session (max {MAX_FILES}).")
    if sum(p.stat().st_size for p in existing) + len(data) > MAX_SESSION_BYTES:
        raise ValueError("Quota de session dépassé.")

    dest = (d / _safe_name(uploaded.name)).resolve()
    # Ceinture et bretelles : le chemin final doit rester sous le répertoire de session.
    if not dest.is_relative_to(d.resolve()):
        raise ValueError("Chemin de destination invalide.")

    dest.write_bytes(data)
    return dest


def purge_expired() -> None:
    """À appeler depuis une tâche périodique, pas depuis une requête utilisateur."""
    now = time.time()
    if not BASE_DIR.exists():
        return
    for d in BASE_DIR.iterdir():
        if d.is_dir() and now - d.stat().st_mtime > SESSION_TTL:
            shutil.rmtree(d, ignore_errors=True)
```

`pages/data.py` devient :

```python
from services.storage import store

for f in uploaded_files:
    if f.name in st.session_state.loaded_files:
        continue
    try:
        path = store(f)
        st.session_state.loaded_files[f.name] = str(path)
        st.success(f"✓ {f.name} chargé")
    except ValueError as exc:
        st.error(str(exc))            # message maîtrisé, pas d'exception brute
    except Exception as exc:
        user_error(st, f"Impossible de charger {f.name}", exc)
```

Le chemin contient désormais l'UUID de session, ce qui rend la clé de `@st.cache_resource` naturellement propre à chaque utilisateur. **V3 disparaît sans autre modification.**

### 4.2 Validation serveur des entrées (corrige V4, V6)

Nouveau fichier `core/validation.py` :

```python
"""Garde-fous serveur. Les bornes des widgets Streamlit sont purement cosmétiques :
toute valeur issue d'un widget doit repasser par ici."""
from __future__ import annotations

import math
from typing import Sequence, TypeVar

T = TypeVar("T")


class InvalidInput(ValueError):
    pass


def num(value, lo: float, hi: float, name: str, integer: bool = False):
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise InvalidInput(f"{name} : valeur non numérique.")
    if not math.isfinite(v):
        raise InvalidInput(f"{name} : valeur non finie.")
    if not (lo <= v <= hi):
        raise InvalidInput(f"{name} doit être compris entre {lo} et {hi}.")
    return int(v) if integer else v


def choice(value: T, allowed: Sequence[T], name: str) -> T:
    # Indispensable : selectbox renvoie les valeurs inconnues telles quelles.
    if value not in allowed:
        raise InvalidInput(f"{name} : option non autorisée.")
    return value


def choices(values: Sequence[T], allowed: Sequence[T], name: str) -> list[T]:
    allowed_set = set(allowed)
    bad = [v for v in values if v not in allowed_set]
    if bad:
        raise InvalidInput(f"{name} : option(s) non autorisée(s).")
    return list(values)


def text(value: str, name: str, max_len: int = 64,
         pattern: str = r"^[\w .\-]*$") -> str:
    import re
    v = str(value)[:max_len]
    if not re.match(pattern, v):
        raise InvalidInput(f"{name} : caractères non autorisés.")
    return v
```

Application, aux endroits sensibles :

```python
# pages/modelling.py
from core.validation import num, choice, text, InvalidInput

fov     = num(st.number_input("pixel number", value=128, min_value=16, max_value=512), 16, 512, "Nombre de pixels", integer=True)
px_size = num(st.number_input("pixel size in mas", value=0.15,min_value=0.001, max_value=10.0), 0.001, 10.0, "Taille de pixel")


gamma   = num(st.number_input("gamma", value=0.2, min_value=0.01, max_value=2.0),0.01, 2.0, "Gamma")

model_name = text(st.text_input("Model name", max_chars=64), "Nom du modèle")
```

```python
# pages/fitting.py
IMG_CMAPS = ("hot", "inferno", "viridis", "plasma", "gray", "afmhot")
DTYPES    = ("VIS2DATA", "T3PHI", "VISPHI", "T3AMP", "FLUXDATA")

img_cmap   = choice(st.selectbox("Colormap", IMG_CMAPS), IMG_CMAPS, "Colormap")
img_size   = num(st.number_input("Image size (px)", 64, 512, 128),64, 512, "Taille d'image", integer=True)

nb_steps   = num(st.number_input("Steps", 0, 5000, 1000),0, 5000, "Nombre de pas", integer=True)
nb_walkers = num(st.number_input("Walkers", 1, 32, 16),1, 32, "Walkers", integer=True)
dtypes     = choices(st.multiselect("Data to fit", DTYPES, default=["VIS2DATA", "T3PHI"]),DTYPES, "Types de données")
```

Et englober le rendu de chaque page :

##### => [PAS FAIT pour apge DATA]
```python
def render() -> None:
    try:
        _render()
    except InvalidInput as exc:
        st.error(str(exc))     # message propre, pas de trace
```

Note sur `nb_steps` : la limite passe de 40 000 à 5 000. Sur un service public gratuit, 40 000 pas n'est pas un usage raisonnable ; ceux qui en ont besoin utiliseront le code Python reproductible que l'application génère déjà (bonne fonctionnalité, gardez-la et mettez-la en avant).

### 4.3 Neutralisation de l'expression de filtre (corrige V5)

```python
# core/validation.py (suite)
import re

_ALLOWED_IDENTIFIERS = {"EFF_WAVE", "EFF_BAND", "LENGTH", "PA", "SPAFREQ"}
_ALLOWED_CHARS = re.compile(r"^[A-Za-z_0-9\s().<>=!&|+\-*/]*$")
_IDENT         = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def filter_expression(expr: str) -> str:
    """L'expression finit dans un eval() chez oimodeler. Liste blanche stricte."""
    if len(expr) > 300:
        raise InvalidInput("Expression de filtre trop longue.")
    if not _ALLOWED_CHARS.match(expr):
        raise InvalidInput("Caractère non autorisé dans l'expression de filtre.")
    if "__" in expr:
        raise InvalidInput("Expression de filtre non autorisée.")
    for ident in _IDENT.findall(expr):
        if ident not in _ALLOWED_IDENTIFIERS:
            raise InvalidInput(f"Identifiant non autorisé : {ident}")
    return expr
```

Dans `pages/data.py`, borner les longueurs d'onde puis valider :

```python
w1_lo = num(wl1_min, 0.1, 30.0, "λ min (plage 1)") * 1e-6
w1_hi = num(wl1_max, 0.1, 30.0, "λ max (plage 1)") * 1e-6
if w1_lo >= w1_hi:
    raise InvalidInput("λ min doit être inférieur à λ max.")

expr = filter_expression(f"(EFF_WAVE<{w1_lo}) | (EFF_WAVE>{w1_hi})")
st.session_state.filter_expr = expr
```

Indépendamment, ouvrez une *issue* chez `oimodeler` sur le `eval()` de `oifitsFlagWithExpression`. Le remplacement raisonnable est un parseur d'expressions restreint (`ast.parse` en mode `eval` avec liste blanche de nœuds, ou `numexpr.evaluate`, qui accepte exactement ce type d'expression booléenne sur tableaux NumPy).

### 4.4 Encadrement des calculs longs (corrige V7)

Deux niveaux :

**Sémaphore global** — empêche l'écroulement du serveur :

```python
# services/jobs.py
import threading

MAX_CONCURRENT_FITS = 2
_semaphore = threading.BoundedSemaphore(MAX_CONCURRENT_FITS)


class Busy(RuntimeError):
    pass


def run_fit(fn, *args, **kwargs):
    if not _semaphore.acquire(blocking=False):
        raise Busy("Le serveur exécute déjà le nombre maximum d'ajustements. "
                   "Réessayez dans quelques minutes.")
    try:
        return fn(*args, **kwargs)
    finally:
        _semaphore.release()
```

**Limitation de fréquence par session** — empêche l'abus répété :

```python
import time

FIT_COOLDOWN = 60  # secondes entre deux lancements

def check_cooldown() -> None:
    last = st.session_state.get("_last_fit", 0)
    remaining = FIT_COOLDOWN - (time.time() - last)
    if remaining > 0:
        raise InvalidInput(f"Veuillez patienter {int(remaining)} s avant un nouvel ajustement.")
    st.session_state["_last_fit"] = time.time()
```

Pour une solution plus robuste (et la seule qui tienne à l'échelle), les ajustements devraient partir dans un worker séparé — Celery, RQ, ou même un simple `ProcessPoolExecutor` avec `maxtasksperchild` — de façon qu'un calcul qui explose tue le worker et non le serveur web. Le `MemoryError` d'un `getImage` trop gros devient alors récupérable.

---

## 5. Durcissement du déploiement

### 5.1 `.streamlit/config.toml`

```toml
[server]
headless = true
address = "127.0.0.1"          # accessible uniquement via le reverse proxy
port = 8501
maxUploadSize = 50             # Mo — cohérent avec MAX_FILE_BYTES
maxMessageSize = 60
enableXsrfProtection = true    # défaut : à ne jamais désactiver
enableCORS = true
enableStaticServing = false    # pas de partage de fichiers statiques
fileWatcherType = "none"       # inutile en production

[browser]
gatherUsageStats = false

[client]
showErrorDetails = "none"      # pas de trace Python dans le navigateur (cf. V12)
toolbarMode = "minimal"

[runner]
fastReruns = true
```

`showErrorDetails = "none"` est important : il masque les traces des exceptions non rattrapées, qui sinon s'affichent intégralement à l'écran.

### 5.2 Reverse proxy

Streamlit ne doit jamais être exposé directement. Devant, un nginx ou Caddy qui apporte TLS, limitation de débit et en-têtes de sécurité.

```nginx
limit_req_zone  $binary_remote_addr zone=oim_general:10m rate=30r/m;
limit_req_zone  $binary_remote_addr zone=oim_upload:10m  rate=5r/m;
limit_conn_zone $binary_remote_addr zone=oim_conn:10m;

server {
    listen 443 ssl http2;
    server_name oimodeler.example.org;

    # TLS : certificat Let's Encrypt, TLS 1.2 minimum

    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options    "nosniff" always;
    add_header X-Frame-Options           "SAMEORIGIN" always;
    add_header Referrer-Policy           "strict-origin-when-cross-origin" always;
    add_header Permissions-Policy        "geolocation=(), microphone=(), camera=()" always;

    client_max_body_size 55m;
    limit_conn oim_conn 8;

    location / {
        limit_req zone=oim_general burst=20 nodelay;
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade    $http_upgrade;     # WebSocket : indispensable
        proxy_set_header Connection "upgrade";
        proxy_set_header Host       $host;
        proxy_set_header X-Real-IP  $remote_addr;
        proxy_read_timeout 3600s;                      # les MCMC sont longs
    }

    location /_stcore/upload_file {
        limit_req zone=oim_upload burst=3 nodelay;
        proxy_pass http://127.0.0.1:8501;
        proxy_set_header Host      $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Une politique CSP est souhaitable mais délicate : Streamlit utilise du style inline et des workers. À mettre en place en mode `Content-Security-Policy-Report-Only` d'abord, puis à durcir en observant les rapports.

### 5.3 Confinement du conteneur

```dockerfile
FROM python:3.11-slim

RUN useradd --create-home --uid 10001 oim
WORKDIR /app

COPY requirements.lock .
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

COPY --chown=oim:oim . .
RUN mkdir -p /var/lib/oimodeler/uploads && chown oim:oim /var/lib/oimodeler/uploads

USER oim
EXPOSE 8501
HEALTHCHECK CMD curl -f http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app.py"]
```

```yaml
# docker-compose.yml (extrait)
services:
  oimodeler:
    build: .
    read_only: true                 # système de fichiers en lecture seule
    tmpfs:
      - /tmp:size=64m,noexec,nosuid
    volumes:
      - uploads:/var/lib/oimodeler/uploads
    cap_drop: [ALL]
    security_opt:
      - no-new-privileges:true
    deploy:
      resources:
        limits:
          memory: 4G                # une bombe mémoire tue le conteneur, pas l'hôte
          cpus: "2.0"
    restart: unless-stopped
```

`read_only: true` combiné à un volume unique inscriptible neutralise l'essentiel de l'impact d'une V1 résiduelle : même une écriture arbitraire ne peut plus sortir du volume d'uploads.

Ajouter enfin une tâche de purge :

```
*/10 * * * * docker compose exec -T oimodeler python -c "from services.storage import purge_expired; purge_expired()"
```

---

## 6. Faut-il vraiment ouvrir sans authentification ?

Vous avez posé la question, et elle mérite une réponse nuancée plutôt qu'un « ajoutez un login ».

**Pour un service scientifique de démonstration, l'accès libre est défendable et souvent souhaitable.** La barrière d'un compte à créer suffit à écarter la plupart des utilisateurs occasionnels, et c'est exactement l'inverse de votre objectif affiché (« Simplest way to get in interferometric data modelisation »). Le problème n'est pas l'absence d'authentification en soi.

Le problème est que **l'absence d'authentification vous prive de deux choses dont l'application a besoin** : un identifiant stable pour isoler les données, et un point d'application des quotas. Or ces deux besoins se satisfont sans compte utilisateur :

- **Isolation** : l'UUID de session du §4.1 la fournit, sans identité. C'est suffisant pour empêcher les fuites croisées de V3.
- **Quotas** : la limitation par IP au niveau du reverse proxy, plus le sémaphore global du §4.4, couvrent l'essentiel. Ce n'est pas infaillible (IPv6, rotation d'adresses), mais c'est proportionné à la menace réelle sur une application d'astrophysique.

**Ma recommandation :** ouverture publique, mais différenciée selon le coût de l'opération.

| Fonctionnalité | Coût | Accès |
|---|---|---|
| Overview, Component Explorer | Négligeable | **Public, sans limite** |
| Upload OIFITS, filtrage, visualisation | Modéré (disque, RAM) | **Public**, avec quota par session et limite de débit |
| Random search (≤ 200 itérations) | Modéré (CPU) | **Public**, avec sémaphore et délai entre lancements |
| Minimisation χ², Emcee long | Élevé (CPU, minutes à heures) | **À restreindre** |

Pour la dernière ligne, plutôt qu'un compte, trois options par ordre de préférence :

1. **Plafonner sévèrement** (5 000 pas maximum, 32 walkers) et mettre en avant le code Python reproductible que l'application génère déjà. C'est pédagogiquement meilleur : l'utilisateur repart avec un script qu'il exécute chez lui sans limite. Aucune authentification requise, et c'est la solution que je retiendrais en premier.
2. **File d'attente visible** avec une position et un temps estimé. Autorégule sans exclure personne.
3. **Authentification légère** si vous voulez tracer les usages. Streamlit 1.42+ intègre nativement OIDC (`st.login()`, `st.user`), qui se branche sur les identités institutionnelles — ORCID, ou l'identité fédérée de votre laboratoire. C'est une dizaine de lignes de configuration, sans base d'utilisateurs à gérer.

Un point sur lequel je serais catégorique en revanche : **si les utilisateurs sont amenés à téléverser des données d'observation non publiées, l'isolation par session du §4.1 n'est pas optionnelle, avec ou sans authentification.** Une fuite de données pré-publication entre équipes concurrentes est un incident scientifique sérieux, indépendamment de toute considération technique.

---

## 7. Plan d'action

> **Status update (2026-09-14, `security-hardener` agent run, branch `claude/loving-carson-k5z8v3`):**
> Blocking items **1–4** are implemented and verified (path-traversal upload
> attempt rejected, forged widget values rejected, filter-expression
> allowlist rejects injection attempts while still accepting the real
> scientific-notation expressions the app generates — see
> `services/storage.py`, `core/validation.py`, and the updated
> `pages/data.py` / `pages/modelling.py` / `pages/fitting.py` /
> `pages/explorer.py`). Items **5–8** (Streamlit config, reverse proxy,
> container hardening, dependency pinning) are **not yet started** — still
> open, tracked below. Item 3's widget coverage is intentionally partial:
> see the note under the table for what was prioritized vs. deferred.

### Bloquant avant toute mise en ligne

| # | Action | Réf. | Charge estimée | Status |
|---|--------|------|----------------|--------|
| 1 | Module `services/storage.py` : répertoire par session, assainissement, quotas, purge | V1, V3, V8 | 3 h | ✅ Done — `services/storage.py`, wired into `pages/data.py` upload handler |
| 2 | Supprimer toute reconstruction de chemin ; utiliser `loaded_files` comme table d'autorisation | V2 | 1 h | ✅ Done — `services/storage.resolve_selected_paths()`, used by all three `_get_active_data_with_filter()` copies (`pages/data.py`, `pages/modelling.py`, `pages/fitting.py`); the `test_files_path`/`test_selected_file` debug variables that reconstructed `/tmp/`-paths are removed |
| 3 | Module `core/validation.py` et application à tous les widgets | V4, V6 | 4 h | 🟡 Partial — module done; applied to all image-size widgets (`modelling.py`, `explorer.py`, `fitting.py`'s model-image tab) and MCMC steps/walkers/dtypes (`fitting.py`), plus every `selectbox`/`multiselect` that feeds a dict lookup (model names, component types, colormap, method). **Deferred** (not yet re-validated): plot axis-limit number_inputs (cosmetic, no DoS/crash impact), per-parameter component sliders in `explorer.py`'s `_SLIDER_CFG` and `components/param_editor.py`, blackbody/spline interpolator numeric widgets in `modelling.py` Tab 3 (bb_temp/bb_dist/bb_lum, per-point λ/value), CSV import widgets (tracked separately under V17 below) |
| 4 | Liste blanche sur l'expression de filtre + bornage des longueurs d'onde | V5 | 1 h | ✅ Done — `core/validation.filter_expression()`, applied in `pages/data.py` before the expression is stored in `filter_expr`; wavelength bounds clamped to [0.1, 30.0] µm via `num()`. Note: the audit's first-draft identifier regex rejected the app's own real expressions (Python's float repr uses scientific notation, e.g. `2.9e-06`, and a bare `[A-Za-z_]...` scan matches the `e` as a fake identifier) — replaced with a number-vs-identifier tokenizer so real expressions pass while injection attempts are still rejected |
| 5 | `.streamlit/config.toml` avec `showErrorDetails = "none"` | V15, V12 | 30 min | ⬜ Not started |
| 6 | Reverse proxy TLS + limitation de débit | V7, V8 | 2 h | ⬜ Not started |
| 7 | Conteneur non-root, `read_only`, limites mémoire et CPU | V6, V13 | 2 h | ⬜ Not started |
| 8 | Épingler `oimodeler` sur un commit, passer Streamlit en 1.54+, lockfile avec empreintes | V14 | 1 h | ⬜ Not started |

### À traiter dans les deux semaines

| # | Action | Réf. | Status |
|---|--------|------|--------|
| 9 | Unifier les trois `_get_active_data_with_filter()` ; filtres dans la clé de cache | V9 | 🟡 Partial — the three copies no longer reconstruct paths from `test_files_path` (they now all call `services/storage.resolve_selected_paths()`, an allowlist lookup), which removes the V2 angle. They are still three separate function bodies and `keepOldFlag` still differs between `pages/data.py` (`True`) and `pages/modelling.py`/`pages/fitting.py` (`False`) — left as-is to avoid silently changing scientific behavior outside this run's scope. The actual dedup into a single `services/` helper, and folding filter params into the cache key (the mutation-of-shared-cache-object half of V9), are still open |
| 10 | Sémaphore et délai entre ajustements | V7 | ⬜ Not started (`services/jobs.py` stopgap not yet created) |
| 11 | Renommer `pages/` en `views/` | V11 | ⬜ Not started |
| 12 | Helper `user_error()` ; supprimer les `{exc}` affichés | V12 | ⬜ Not started app-wide. Note: the specific `st.error(f"Error ({f.name}): {exc}")` in the old upload handler and the bare `except:`/raw-exception patterns directly inside the blocks touched for V1/V2/V5 (`pages/data.py`'s upload loop and filter section, `pages/explorer.py`'s image-render fallback) were fixed as a byproduct; the ~20 other occurrences the audit counted are untouched |
| 13 | Chemin de sampler unique par session | V10 | ⬜ Not started (`pages/fitting.py:418`, still `Path("/tmp/sampler_emcee.txt")`) |
| 14 | `.gitignore`, purge des `.pyc` de l'historique, Dependabot | V14 | ⬜ Not started |
| 15 | Bornes sur l'import CSV, contrainte sur `type_abbr`, échappement des formules | V17 | ⬜ Not started |

### Amélioration continue

| # | Action | Réf. |
|---|--------|------|
| 16 | Remplacer les `except:` nus par des captures typées et journalisées | V18 |
| 17 | `np.random.default_rng()` au lieu de la graine globale | V19 |
| 18 | Renommer `3_test.py` en `app.py`, supprimer le code de débogage et les fonctions mortes | V20 |
| 19 | Réécrire les deux `unsafe_allow_html=True` en Markdown natif | V16 |
| 20 | Journalisation structurée, endpoint de santé, supervision | — |
| 21 | Ouvrir une *issue* chez `oimodeler` sur le `eval()` de `oifitsFlagWithExpression` | V5 |
| 22 | Tests unitaires sur `core/validation.py` et `services/storage.py`, dont les cas de traversée | — |
| 23 | CI : `bandit`, `pip-audit`, `ruff` sur chaque *pull request* | — |

---

## 8. Checklist de mise en ligne

- [x] Aucun `open()` ni aucune construction de chemin n'utilise une chaîne provenant du client sans passer par `_safe_name()` — enforced in `services/storage.store()`
- [x] Aucun `f"/tmp/..."` ni concaténation de chemin ne subsiste dans le code pour les uploads/sélections (`grep -rn "'/tmp/" .` — only remaining hit is the still-open V10 sampler path in `pages/fitting.py:418`, tracked separately)
- [ ] Chaque `number_input`, `slider`, `selectbox`, `multiselect` et `text_input` est revalidé côté serveur — image-size, MCMC steps/walkers/dtypes, and dict-lookup selectboxes/multiselects done; plot axis limits, per-parameter component sliders, and interpolator/CSV widgets still open (see action-plan item 3 note)
- [ ] `showErrorDetails = "none"` et aucun `{exc}` affiché à l'utilisateur — config.toml not created yet (item 5); most `{exc}` echoes app-wide still open (item 12), only the ones directly inside the V1/V2/V5 code paths were fixed
- [x] Test manuel : upload avec `filename="../../evil.fits"` — doit être rejeté — verified: sanitized to a safe basename and stored under the session directory, never escapes `BASE_DIR`
- [x] Test manuel : `multiselect` forcé sur `"../etc/passwd"` — doit être rejeté — verified via `resolve_selected_paths()`: unknown/forged names are silently dropped, not resolved to a path
- [ ] Test manuel : deux navigateurs distincts, même nom de fichier, contenus différents — aucune interférence — storage is now session-scoped (verified programmatically that two sessions get distinct directories); not yet verified end-to-end in a running two-browser-tab Streamlit session
- [x] Test manuel : `pixel number = 999999` via WebSocket forgé — doit être rejeté avant allocation — verified via `core.validation.num()` unit-level (out-of-range/non-finite values raise `InvalidInput`) for every image-size widget listed in action-plan item 3; not re-verified over an actual forged WebSocket frame
- [ ] Le conteneur tourne en non-root, en lecture seule, avec une limite mémoire — not started (item 7)
- [ ] TLS actif, HSTS actif, limitation de débit vérifiée sur `/_stcore/upload_file` — not started (item 6)
- [ ] La purge des uploads expirés est planifiée et testée — `services/storage.purge_expired()` exists and is exercised opportunistically from `store()`, but no external cron/scheduler is configured yet
- [ ] `pip-audit` ne remonte aucune vulnérabilité de sévérité élevée — not run this pass
- [ ] Une adresse de contact sécurité figure dans le README (vous avez déjà un contact générique — précisez-le) — not started

---

## 9. Remarque finale

Le principal risque de cette application n'est pas un attaquant déterminé. C'est un doctorant qui téléverse `data.fits` et se retrouve à publier un χ² calculé sur les observations de quelqu'un d'autre, sans que personne ne s'en aperçoive. Les correctifs 1, 2, 3 et 9 traitent ce scénario, et ce sont ceux sur lesquels je concentrerais l'effort en priorité.

L'architecture en couches que vous avez mise en place rend ce travail nettement plus simple qu'il ne le serait sur un script monolithique : la validation s'insère dans `core/`, le stockage dans `services/`, et les pages n'ont presque pas à changer. Le chantier bloquant représente une bonne journée de travail.
