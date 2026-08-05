# Deployment

Backend → **Render** (Docker). Frontend → **Vercel** (static build).

The two deployments depend on each other, so the order matters:

1. Deploy the backend and copy its URL.
2. Deploy the frontend with that URL as an environment variable.
3. Go back to Render and allow the frontend's URL through CORS.

Step 3 is the one people skip. Without it the site loads but every request fails.

---

## Before you start

Push the repo to GitHub with this structure:

```
your-repo/
├── backend/          Dockerfile lives here
├── frontend/
├── render.yaml
└── DEPLOY.md
```

Check that `backend/models/` is **empty** in the repo — model files are gitignored on purpose. The Dockerfile trains the model during the image build, so nothing needs committing. (See "About the model" at the end.)

---

## Part 1 — Backend on Render

### Option A: Blueprint (recommended)

`render.yaml` at the repo root already describes the service.

1. Render Dashboard → **New** → **Blueprint**
2. Connect your GitHub repo → Render reads `render.yaml` → **Apply**
3. Leave the environment variables blank for now; you'll fill them in Part 3.

### Option B: Manual

Render Dashboard → **New** → **Web Service** → connect the repo, then:

| Field | Value |
|---|---|
| Language / Runtime | **Docker** |
| Root Directory | `backend` |
| Dockerfile Path | `./Dockerfile` |
| Health Check Path | `/health` |
| Instance Type | Free |

**Root Directory must be `backend`.** The Dockerfile is not at the repo root, and if you leave this blank the build fails immediately with "Dockerfile not found".

### Check it worked

The first build takes 3–6 minutes: it installs dependencies, builds the dataset, and trains the model. You'll see this in the logs:

```
[data] no .../data/raw/reviews.csv found -- generating a bootstrap sample.
[train] saved model -> /app/models/baseline.joblib
```

Then visit:

```
https://<your-service>.onrender.com/health
→ {"status":"ok","model_loaded":true,"model_backend":"baseline",...}
```

`"model_loaded": true` is the thing to check. If it says `false`, the training step didn't produce a model — read the build log rather than the runtime log.

Interactive API docs: `https://<your-service>.onrender.com/docs`

**Copy this URL.** You need it in Part 2.

---

## Part 2 — Frontend on Vercel

1. Vercel → **Add New** → **Project** → import the repo
2. Set **Root Directory** to `frontend` (click Edit next to it — this is the second easy thing to get wrong)
3. Framework Preset: **Vite** (usually auto-detected)
4. Expand **Environment Variables** and add both:

| Name | Value |
|---|---|
| `VITE_API_BASE_URL` | `https://<your-service>.onrender.com` |
| `VITE_USE_MOCK` | `false` |

5. **Deploy**

Two things about `VITE_API_BASE_URL`:

- No trailing slash and no `/api` suffix — just the origin. (A trailing slash is stripped in code, but the `/api` suffix would break every route.)
- Vite bakes `VITE_*` variables into the bundle **at build time**, not at runtime. Change one later and you must redeploy for it to take effect. Changing it in the dashboard alone does nothing to the already-built site.

If you skip `VITE_USE_MOCK=false`, the site deploys and works — on fake data. It looks convincing, which is exactly why it's worth double-checking.

---

## Part 3 — Connect them (CORS)

The API rejects browser requests from origins it doesn't know. Right now it doesn't know your Vercel URL.

In Render → your service → **Environment** → add:

| Key | Value |
|---|---|
| `ALLOWED_ORIGINS` | `https://your-app.vercel.app` |

Multiple origins are comma-separated, no spaces:
`https://your-app.vercel.app,https://custom-domain.com`

**For Vercel preview deployments**, every commit gets its own subdomain, so listing them is impossible. Add this instead:

| Key | Value |
|---|---|
| `ALLOWED_ORIGIN_REGEX` | `https://.*-your-team\.vercel\.app` |

Replace `your-team` with the slug from an actual preview URL. Keep the pattern narrow — `https://.*\.vercel\.app` would allow every Vercel site on the internet to call your API.

Render redeploys automatically when you save an environment variable. Wait for it to go live, then load your Vercel site and analyze a review.

### If it still fails

Open the browser console (F12).

| What you see | What it means |
|---|---|
| `No 'Access-Control-Allow-Origin' header` | `ALLOWED_ORIGINS` doesn't exactly match your site's origin. Check `https://` vs `http://`, and no trailing slash. |
| Requests going to `your-app.vercel.app/api/...` | `VITE_API_BASE_URL` wasn't set at build time. Set it, then **redeploy**. |
| `503` with "No model is loaded" | The image built without a model. Check the Render **build** log for the training step. |
| First request hangs ~50s, then works | Normal on the free tier. See below. |

---

## About the free tier

Render's free instances sleep after about 15 minutes of inactivity, and the next request has to wait for a cold start — often 50+ seconds. The frontend allows 75 seconds before timing out and, if it does, says so in plain language rather than showing a generic error.

For a demo or a presentation: **open the API's `/health` URL a minute beforehand** to wake it up. A grader clicking your link cold will otherwise sit and watch a spinner.

Free instances also have limited RAM. The TF-IDF baseline is comfortable there. The DistilBERT model is not — if you train the transformer, expect to move to a paid instance or serve the baseline in production and present the transformer's results from your local evaluation.

---

## About the model that gets deployed

The Dockerfile runs `build_dataset` then `train_baseline` during the image build. This is deliberate: model files are gitignored, so a cloud build cloning your repo would otherwise produce an image with no model in it.

**With no dataset committed, this trains on the synthetic bootstrap sample.** The deployed API will work, respond sensibly, and mean nothing — the metrics from that data are not results. That's fine for showing the system works end to end. It is not fine for a report.

To deploy a real model, either:

- **Commit the dataset.** Put your licensed CSV at `backend/data/raw/reviews.csv`, remove `data/raw/*` from `backend/.gitignore`, and uncomment the `COPY data/raw ./data/raw` line in the Dockerfile. Check the dataset licence permits redistribution before pushing it to a public repo — many Kaggle datasets do not.
- **Or download it in the build.** Add a `RUN curl -o data/raw/reviews.csv <url>` step before training, keeping credentials in build secrets rather than the Dockerfile.

Either way, say plainly in your report which one the deployed instance is running.

---

## Redeploying

| Change | What to do |
|---|---|
| Backend code | Push to the branch — Render auto-deploys |
| Frontend code | Push — Vercel auto-deploys |
| `ALLOWED_ORIGINS` | Save in Render; it redeploys itself |
| `VITE_*` variable | Save in Vercel, then **manually redeploy** — build-time only |
| Retrain the model | Push any backend change; the image rebuilds and retrains |

---

## Checklist before you submit the links

- [ ] `/health` returns `"model_loaded": true`
- [ ] `/docs` loads
- [ ] The Vercel site analyzes a review and shows a real result
- [ ] The dashboard and review queue both render
- [ ] The browser console is free of CORS errors
- [ ] You woke the API up shortly before demoing
- [ ] You can say which dataset the deployed model was trained on
